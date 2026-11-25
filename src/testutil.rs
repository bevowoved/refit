use crate::graph::build_dag;
use crate::hygiene::HygieneScorer;
use crate::mvcc::{CommitOutcome, MvccStore};
use crate::proxy::{CacheMode, ProxyConfig, ProxyState, start_proxy};
use crate::telemetry::TelemetryHub;
use crate::types::{DagPlan, ModuleUnit};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

static TEST_SEQ: AtomicU64 = AtomicU64::new(1);

/// Deterministic SplitMix64 PRNG shared across unit/property tests.
#[derive(Debug, Clone, Copy)]
pub struct TestRng(u64);

impl TestRng {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound.max(1)
    }
}

pub struct TempRepo {
    cleanup: PathBuf,
    pub root: PathBuf,
}

impl TempRepo {
    pub fn new(tag: &str) -> Self {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("refit_{tag}_{}_{seq}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self { cleanup: dir.clone(), root: dir }
    }

    pub fn nested(tag: &str, child: &str) -> Self {
        let mut t = Self::new(tag);
        t.root = t.cleanup.join(child);
        fs::create_dir_all(&t.root).unwrap();
        t
    }

    pub fn write(&self, rel: &str, body: impl AsRef<[u8]>) -> &Self {
        let p = self.root.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, body).unwrap();
        self
    }

    pub fn write_all(&self, files: &[(&str, &str)]) -> &Self {
        for &(rel, body) in files {
            self.write(rel, body);
        }
        self
    }

    pub fn with_files(tag: &str, files: &[(&str, &str)]) -> Self {
        let t = Self::new(tag);
        t.write_all(files);
        t
    }

    pub fn nested_with(tag: &str, child: &str, files: &[(&str, &str)]) -> Self {
        let t = Self::nested(tag, child);
        t.write_all(files);
        t
    }

    pub fn git(tag: &str, child: &str, files: &[(&str, &str)]) -> Self {
        let repo = Self::nested_with(tag, child, files);
        let run = |args: &[&str]| {
            let mut cmd = Command::new("git");
            cmd.args(args).current_dir(repo.path());
            let out = cmd.output().expect("git");
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(out.status.success(), "git {args:?}: {err}");
        };
        for args in [
            &["init", "-b", "main"][..],
            &["config", "user.email", "refit@local.test"],
            &["config", "user.name", "Refit Test"],
            &["add", "-A"],
            &["commit", "-m", "init"],
        ] {
            run(args);
        }
        repo
    }

    pub fn path(&self) -> &Path { &self.root }
    pub fn join(&self, rel: &str) -> PathBuf { self.root.join(rel) }
    pub fn has(&self, rel: &str) -> bool { self.join(rel).exists() }
    pub fn read(&self, rel: &str) -> String { fs::read_to_string(self.join(rel)).unwrap() }
    pub fn hash(&self, rel: &str) -> String { MvccStore::hash_file(&self.join(rel)).unwrap() }
}

impl Drop for TempRepo {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.cleanup); }
}

pub fn unit_stub(id: &str, files: &[&str]) -> ModuleUnit {
    ModuleUnit {
        id: id.into(),
        files: files.iter().map(|&f| f.to_string()).collect(),
        layer: 0, wave: 0, deps: vec![], dependents: vec![], test_files: vec![],
        loc: 1, base_blake3: BTreeMap::new(), abi_hash: BTreeMap::new(),
        baseline_hygiene: 0.90, allow_abi_change: false,
    }
}

pub fn map1<V>(k: &str, v: V) -> BTreeMap<String, V> {
    BTreeMap::from([(k.to_string(), v)])
}

pub fn asrf_edit(path: &str, search: &str, replace: &str) -> String {
    format!(
        "<<<REFIT_PATCH path=\"{path}\">>>\n<<<SEARCH>>>\n{search}\n\
         <<<REPLACE>>>\n{replace}\n<<<END>>>"
    )
}

pub fn plan_repo(root: &Path, scorer: &HygieneScorer) -> DagPlan {
    build_dag(root, None, scorer, None).unwrap()
}

pub fn commit_file(
    store: &MvccStore,
    root: &Path,
    mid: &str,
    rel: &str,
    body: impl AsRef<[u8]>,
) -> (u64, String, String) {
    let base = MvccStore::hash_file(&root.join(rel)).unwrap();
    let res = store.commit_tx(mid, &map1(rel, base), &map1(rel, body.as_ref().to_vec()), &[]).unwrap();
    match res {
        CommitOutcome::Committed { head_rev, tx_id, new_hashes } => {
            (head_rev, tx_id, new_hashes[rel].clone())
        }
        other => panic!("expected commit, got {other:?}"),
    }
}

pub fn sse_delta(content: &str) -> Vec<u8> {
    let payload = json!({ "choices": [{ "delta": { "content": content } }] });
    format!("data: {payload}\n\n").into_bytes()
}

pub fn http_res(status: &str, extra: &str, body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 {status}\r\n{extra}Content-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

pub fn proxy_cfg(upstream: &str, retention: CacheMode) -> ProxyConfig {
    let mut cfg = ProxyConfig::new(
        "127.0.0.1:0".into(), upstream.into(), "test-key".into(),
        "moonshotai-kimi-k3.1-pro-trial".into(),
    );
    cfg.cache_retention = retention;
    cfg.preserve_tools = false;
    cfg
}

async fn bind_ephemeral() -> TcpListener {
    TcpListener::bind("127.0.0.1:0").await.unwrap()
}

pub struct TestProxy {
    pub addr: String,
    pub state: Arc<ProxyState>,
    pub client: reqwest::Client,
    auth: Option<String>,
    handle: JoinHandle<()>,
    _tmp: TempRepo,
}

impl TestProxy {
    pub async fn spawn(upstream: &str, retention: CacheMode, token: Option<&str>) -> Self {
        let l = bind_ephemeral().await;
        let addr = l.local_addr().unwrap().to_string();
        drop(l);

        let tmp = TempRepo::new("prx");
        let tel = TelemetryHub::new(tmp.join("run.jsonl")).unwrap();
        let mut cfg = proxy_cfg(upstream, retention);
        cfg.listen_addr = addr.clone();
        cfg.auth_token = token.map(String::from);

        let (state, handle) = start_proxy(cfg, tel).await.unwrap();
        let auth = token.map(String::from);
        Self { addr, state, client: reqwest::Client::new(), auth, handle, _tmp: tmp }
    }

    pub fn url(&self, path: &str) -> String { format!("http://{}{path}", self.addr) }

    async fn send_json(&self, mut req: reqwest::RequestBuilder) -> (u16, Value) {
        if let Some(ref tok) = self.auth {
            req = req.bearer_auth(tok);
        }
        let res = req.send().await.unwrap();
        (res.status().as_u16(), res.json().await.unwrap_or(Value::Null))
    }

    pub async fn get_json(&self, path: &str) -> (u16, Value) {
        self.send_json(self.client.get(self.url(path))).await
    }

    pub async fn post_json(&self, path: &str, body: &Value) -> (u16, Value) {
        self.send_json(self.client.post(self.url(path)).json(body)).await
    }
}

impl Drop for TestProxy {
    fn drop(&mut self) { self.handle.abort(); }
}

pub struct MockHttp {
    pub addr: String,
    calls: Arc<AtomicU64>,
    handle: JoinHandle<()>,
}

impl MockHttp {
    pub async fn spawn(resps: Vec<Vec<u8>>) -> Self {
        let up = bind_ephemeral().await;
        let addr = up.local_addr().unwrap().to_string();
        let calls = Arc::new(AtomicU64::new(0));
        let calls_c = calls.clone();

        let handle = tokio::spawn(async move {
            for payload in resps {
                let Ok((mut sock, _)) = up.accept().await else { break };
                let _ = sock.read(&mut [0u8; 4096]).await;
                calls_c.fetch_add(1, Ordering::SeqCst);
                let _ = sock.write_all(&payload).await;
                let _ = sock.shutdown().await;
            }
        });
        Self { addr, calls, handle }
    }

    pub fn calls(&self) -> u64 { self.calls.load(Ordering::SeqCst) }
    pub fn url(&self, path: &str) -> String { format!("http://{}{path}", self.addr) }
}

impl Drop for MockHttp {
    fn drop(&mut self) { self.handle.abort(); }
}

/// Ephemeral local Axum HTTP server for integration tests (auto-aborts on drop).
pub struct TestAxum {
    pub base_url: String,
    handle: JoinHandle<()>,
}

impl TestAxum {
    pub async fn spawn(app: axum::Router) -> Self {
        let listener = bind_ephemeral().await;
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Self { base_url, handle }
    }
}

impl Drop for TestAxum {
    fn drop(&mut self) { self.handle.abort(); }
}

pub fn assert_contains_all(haystack: &str, needles: &[&str]) {
    for &needle in needles {
        assert!(haystack.contains(needle), "missing expected `{needle}` in:\n{haystack}");
    }
}

pub async fn mock_sse(chunks: Vec<Vec<u8>>, delay_ms: u64, hold_secs: u64) -> String {
    let listener = bind_ephemeral().await;
    let addr = listener.local_addr().unwrap().to_string();

    tokio::spawn(async move {
        let Ok((mut sock, _)) = listener.accept().await else { return };
        let _ = sock.read(&mut [0u8; 4096]).await;
        let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                   Transfer-Encoding: chunked\r\n\r\n";
        let _ = sock.write_all(hdr.as_bytes()).await;

        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 && delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            let _ = sock.write_all(format!("{:x}\r\n", chunk.len()).as_bytes()).await;
            let _ = sock.write_all(chunk).await;
            let _ = sock.write_all(b"\r\n").await;
            let _ = sock.flush().await;
        }
        if hold_secs > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(hold_secs)).await;
        } else {
            let _ = sock.write_all(b"0\r\n\r\n").await;
            let _ = sock.flush().await;
        }
    });
    addr
}

pub fn nest_src(head: &str, step: &str, open: &[String], body: &str, close: Option<&str>) -> String {
    let mut out = format!("{head}\n");
    for (i, line) in open.iter().enumerate() {
        out.push_str(&format!("{}{line}\n", step.repeat(i + 1)));
    }
    out.push_str(&format!("{}{body}\n", step.repeat(open.len() + 1)));
    if let Some(ch) = close {
        for i in (0..open.len()).rev() {
            out.push_str(&format!("{}{ch}\n", step.repeat(i + 1)));
        }
        out.push_str("  return 0;\n}\n");
    }
    out
}

pub fn repeat_fn(
    head: &str,
    range: std::ops::RangeInclusive<usize>,
    mut line: impl FnMut(usize) -> String,
    tail: &str,
) -> String {
    let mut out = format!("{head}\n");
    for i in range {
        out.push_str(&line(i));
    }
    out.push_str(tail);
    out
}

pub fn map_conds(vars: &[&str], fmt: impl Fn(&str) -> String) -> Vec<String> {
    vars.iter().map(|v| fmt(v)).collect()
}

pub fn walk_rs_files(base: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    for entry in fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            walk_rs_files(base, &p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            let stripped = p.strip_prefix(base).unwrap();
            let rel = stripped.to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
}

fn push_parts(out: &mut Vec<u8>, parts: &[&[u8]]) {
    for part in parts {
        out.extend_from_slice(part);
    }
}

pub fn build_single_entry_zip(name: &str, method: u16, comp: &[u8], uncomp_len: u32) -> Vec<u8> {
    let (mut out, nb, clen) = (Vec::new(), name.as_bytes(), comp.len() as u32);
    let nlen = nb.len() as u16;
    let (z16, z32) = (0u16.to_le_bytes(), 0u32.to_le_bytes());
    let (v20, one) = (20u16.to_le_bytes(), 1u16.to_le_bytes());
    let (m_b, cl_b, ul_b, nl_b) = (
        method.to_le_bytes(), clen.to_le_bytes(), uncomp_len.to_le_bytes(), nlen.to_le_bytes(),
    );
    push_parts(&mut out, &[
        &0x0403_4b50u32.to_le_bytes(), &v20, &z16, &m_b, &z32, &z32, &cl_b, &ul_b, &nl_b,
        &z16, nb, comp,
    ]);
    let cd_off = out.len() as u32;
    push_parts(&mut out, &[
        &0x0201_4b50u32.to_le_bytes(), &v20, &v20, &z16, &m_b, &z32, &z32, &cl_b, &ul_b, &nl_b,
        &z16, &z16, &z16, &z16, &z32, &z32, nb,
    ]);
    let cd_sz = (out.len() as u32) - cd_off;
    push_parts(&mut out, &[
        &0x0605_4b50u32.to_le_bytes(), &z16, &z16, &one, &one,
        &cd_sz.to_le_bytes(), &cd_off.to_le_bytes(), &z16,
    ]);
    out
}
