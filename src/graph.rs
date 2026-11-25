mod abi;
mod packet;
mod patch;

pub use abi::hash_abi;
#[cfg(test)]
pub use abi::{check_interface_cutoff, python_abi};
pub use packet::{build_packet, deps_context};
#[cfg(test)]
pub use packet::wave_context;
pub use patch::{AppliedPatch, apply_patches, parse_patches};

use crate::hygiene::HygieneScorer;
use crate::mvcc::MvccStore;
use crate::platform::os_tag;
use crate::source::{count_lines, ext_str, read_text, scan_files, stem_of};
use crate::stem::analysis::{CsrGraph, match_tests_inverted};
use crate::stem::{EdgeScan, scan_edges};
use crate::types::{DagPlan, ModuleUnit};
use anyhow::{Context, Result};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;

type SccWaves = (Vec<Vec<String>>, Vec<BTreeSet<usize>>, Vec<BTreeSet<usize>>, Vec<usize>);

struct BuildCtx<'a> {
    repo_root: &'a Path,
    preloaded: Option<&'a BTreeMap<String, Vec<u8>>>,
    hygiene: &'a HygieneScorer,
    scc_ids: &'a [String],
    mod_tests: &'a HashMap<String, Vec<String>>,
}

pub fn build_dag(
    repo_root: &Path,
    stem_bin_hint: Option<&Path>,
    hygiene: &HygieneScorer,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> Result<DagPlan> {
    let discovery = scan_edges(repo_root, stem_bin_hint, preloaded);
    dag_from(repo_root, discovery, hygiene, preloaded)
}

pub fn dag_from(
    repo_root: &Path,
    discovery: EdgeScan,
    hygiene: &HygieneScorer,
    preloaded: Option<&BTreeMap<String, Vec<u8>>>,
) -> Result<DagPlan> {
    let scan = scan_files(repo_root, preloaded);
    let source_files: Vec<String> = scan.sources.into_iter().map(|(r, _, _)| r).collect();
    let test_files: Vec<String> = scan.tests.into_iter().map(|(r, _, _)| r).collect();

    let mod_tests = match_tests_inverted(repo_root, &source_files, &test_files, preloaded);
    let active_edges = if !discovery.runtime_edges.is_empty() || discovery.edges.is_empty() {
        &discovery.runtime_edges
    } else {
        &discovery.edges
    };
    let (scc_files, scc_deps, scc_dependents, wave_of) =
        compute_waves(&source_files, active_edges);
    let scc_ids = assign_ids(&scc_files);
    let bctx = BuildCtx { repo_root, preloaded, hygiene, scc_ids: &scc_ids, mod_tests: &mod_tests };

    let built: Result<Vec<ModuleUnit>> = if crate::platform::budget().is_constrained() {
        (0..scc_files.len())
            .map(|i| build_unit(&bctx, &scc_files[i], &scc_deps[i], &scc_dependents[i], wave_of[i], i))
            .collect()
    } else {
        (0..scc_files.len())
            .into_par_iter()
            .map(|i| build_unit(&bctx, &scc_files[i], &scc_deps[i], &scc_dependents[i], wave_of[i], i))
            .collect()
    };

    let mut units = built?;
    let total_loc: usize = units.iter().map(|u| u.loc).sum();
    let max_wave = units.iter().fold(0usize, |m, u| m.max(u.wave));
    sort_units(&mut units);

    Ok(DagPlan {
        repo_root: repo_root.display().to_string(),
        host_os: os_tag(),
        stem_used: discovery.used_stem,
        stem_index_ms: discovery.index_ms,
        total_files: source_files.len(),
        total_loc,
        scc_count: units.iter().filter(|u| u.files.len() > 1).count(),
        wave_count: if units.is_empty() { 0 } else { max_wave + 1 },
        units,
    })
}

fn compute_waves(source_files: &[String], edges: &[(String, String)]) -> SccWaves {
    let (csr, csr_edges) = CsrGraph::from_named_edges(source_files, edges);
    let sccs = csr.pearce_scc();

    let mut file_scc = vec![0usize; source_files.len()];
    let mut scc_files = Vec::with_capacity(sccs.len());
    for (idx, comp) in sccs.into_iter().enumerate() {
        let mut files = Vec::with_capacity(comp.len());
        for node in comp {
            let nu = node as usize;
            file_scc[nu] = idx;
            files.push(source_files[nu].clone());
        }
        files.sort();
        scc_files.push(files);
    }

    let n = scc_files.len();
    let mut deps = vec![BTreeSet::new(); n];
    let mut dependents = vec![BTreeSet::new(); n];
    for &(u, v) in &csr_edges {
        let su = file_scc[u as usize];
        let sv = file_scc[v as usize];
        if su != sv {
            deps[su].insert(sv);
            dependents[sv].insert(su);
        }
    }

    let mut wave = vec![0usize; n];
    let mut in_deg: Vec<usize> = deps.iter().map(BTreeSet::len).collect();
    let mut q: VecDeque<usize> = in_deg
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| (d == 0).then_some(i))
        .collect();

    while let Some(u) = q.pop_front() {
        for &v in &dependents[u] {
            wave[v] = wave[v].max(wave[u] + 1);
            in_deg[v] = in_deg[v].saturating_sub(1);
            if in_deg[v] == 0 {
                q.push_back(v);
            }
        }
    }
    (scc_files, deps, dependents, wave)
}

fn scc_label(files: &[String], stem_counts: &HashMap<&str, usize>) -> String {
    if files.len() == 1 {
        let rel = &files[0];
        let st = match stem_of(Path::new(rel)) {
            "" => "mod",
            s => s,
        };
        if stem_counts.get(st).copied().unwrap_or(0) > 1 {
            return rel.rsplit_once('.').map_or_else(|| rel.clone(), |(a, _)| a.to_string());
        }
        return st.to_string();
    }

    let parts: Vec<&str> = files
        .iter()
        .take(4)
        .map(|f| match stem_of(Path::new(f)) {
            "" => "m",
            s => s,
        })
        .collect();
    if files.len() > 4 {
        let h = MvccStore::hash_bytes(files.join(",").as_bytes());
        format!("scc_{}_{}", parts.join("_"), &h[..6])
    } else {
        format!("scc_{}", parts.join("_"))
    }
}

fn assign_ids(scc_files: &[Vec<String>]) -> Vec<String> {
    let mut stem_counts: HashMap<&str, usize> = HashMap::new();
    for files in scc_files.iter().filter(|f| f.len() == 1) {
        let st = match stem_of(Path::new(&files[0])) {
            "" => "mod",
            s => s,
        };
        *stem_counts.entry(st).or_insert(0) += 1;
    }

    let mut ids: Vec<String> = scc_files.iter().map(|f| scc_label(f, &stem_counts)).collect();
    let mut id_counts: HashMap<String, usize> = HashMap::new();
    for id in &ids {
        *id_counts.entry(id.clone()).or_insert(0) += 1;
    }

    for (i, id) in ids.iter_mut().enumerate() {
        if id_counts.get(id).copied().unwrap_or(0) <= 1 {
            continue;
        }
        let files = &scc_files[i];
        if files.len() == 1 {
            *id = files[0].clone();
        } else {
            let h = MvccStore::hash_bytes(files.join(",").as_bytes());
            *id = format!("{id}_{}", &h[..6]);
        }
    }
    ids
}

fn build_unit(
    ctx: &BuildCtx<'_>,
    files: &[String],
    deps: &BTreeSet<usize>,
    dependents: &BTreeSet<usize>,
    wave: usize,
    idx: usize,
) -> Result<ModuleUnit> {
    let (mut base, mut abi, mut tests) = (BTreeMap::new(), BTreeMap::new(), BTreeSet::new());
    let (mut loc, mut score_sum) = (0usize, 0.0f64);

    for rel in files {
        let content = if let Some(txt) = read_text(ctx.repo_root, rel, ctx.preloaded) {
            txt
        } else {
            let abs = ctx.repo_root.join(rel);
            let raw = fs::read_to_string(&abs)
                .with_context(|| format!("read source file {}", abs.display()))?;
            std::borrow::Cow::Owned(raw)
        };
        loc += count_lines(content.as_bytes());
        base.insert(rel.clone(), MvccStore::hash_bytes(content.as_bytes()));
        abi.insert(rel.clone(), hash_abi(rel, &content));
        score_sum += ctx.hygiene.score_content(rel, &content).score;
        if let Some(ts) = ctx.mod_tests.get(rel) {
            tests.extend(ts.iter().cloned());
        }
    }

    Ok(ModuleUnit {
        id: ctx.scc_ids[idx].clone(),
        files: files.to_vec(),
        layer: infer_layer(&files[0], wave),
        wave,
        deps: collect_ids(deps, ctx.scc_ids),
        dependents: collect_ids(dependents, ctx.scc_ids),
        test_files: tests.into_iter().collect(),
        loc,
        base_blake3: base,
        abi_hash: abi,
        baseline_hygiene: score_sum / (files.len().max(1) as f64),
        allow_abi_change: files.len() > 1 && files.len() <= 4,
    })
}

fn collect_ids(set: &BTreeSet<usize>, scc_ids: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(set.len());
    for &d in set {
        out.push(scc_ids[d].clone());
    }
    out
}

fn sort_units(units: &mut [ModuleUnit]) {
    units.sort_by(|a, b| {
        let af = a.files.first().map_or("", String::as_str);
        let bf = b.files.first().map_or("", String::as_str);
        let (ap, bp) = (Path::new(af), Path::new(bf));
        a.wave
            .cmp(&b.wave)
            .then(a.layer.cmp(&b.layer))
            .then(ext_str(ap).cmp(ext_str(bp)))
            .then(ap.parent().cmp(&bp.parent()))
            .then(a.id.cmp(&b.id))
    });
}

fn infer_layer(rel: &str, wave: usize) -> usize {
    if rel.contains("/core/") || rel.starts_with("core/") {
        0
    } else if rel.contains("/engine/") || rel.starts_with("engine/") {
        1
    } else if rel.contains("/services/") || rel.starts_with("services/") {
        2
    } else {
        wave.min(3)
    }
}

pub fn impacted_tests(plan: &DagPlan, changed_files: &[String]) -> Vec<String> {
    let by_id: HashMap<&str, &ModuleUnit> = plan.units.iter().map(|u| (u.id.as_str(), u)).collect();
    let changed: HashSet<String> = changed_files.iter().map(|s| s.replace('\\', "/")).collect();

    let mut q = VecDeque::new();
    let mut visited = HashSet::new();
    for u in &plan.units {
        if u.files.iter().any(|f| changed.contains(f)) && visited.insert(u.id.as_str()) {
            q.push_back(u.id.as_str());
        }
    }

    let mut tests = BTreeSet::new();
    while let Some(uid) = q.pop_front() {
        let Some(u) = by_id.get(uid) else { continue };
        tests.extend(u.test_files.iter().cloned());
        for dep in &u.dependents {
            if visited.insert(dep.as_str()) {
                q.push_back(dep.as_str());
            }
        }
    }
    tests.into_iter().collect()
}

/// Iterative longest-path critical depth over the unit dependent graph (zero recursion).
pub fn critical_depths(units: &[ModuleUnit]) -> HashMap<String, usize> {
    let n = units.len();
    let mut idx_of: FxHashMap<&str, usize> =
        FxHashMap::with_capacity_and_hasher(n, Default::default());
    for (i, u) in units.iter().enumerate() {
        idx_of.insert(u.id.as_str(), i);
    }

    let mut state = vec![0u8; n]; // 0 = unvisited, 1 = on_stack, 2 = done
    let mut depth = vec![0usize; n];
    let mut stack: Vec<(usize, usize)> = Vec::with_capacity(n.min(256));

    for start in 0..n {
        if state[start] != 0 {
            continue;
        }
        state[start] = 1;
        stack.push((start, 0));

        while let Some((u, ei)) = stack.pop() {
            if step_depth_edges(u, ei, units, &idx_of, &mut state, &mut depth, &mut stack) {
                continue;
            }
            state[u] = 2;
            if let Some(&(parent, _)) = stack.last() {
                depth[parent] = depth[parent].max(1 + depth[u]);
            }
        }
    }

    let mut memo = HashMap::with_capacity(n);
    for (i, u) in units.iter().enumerate() {
        memo.insert(u.id.clone(), depth[i]);
    }
    memo
}

fn step_depth_edges(
    u: usize,
    mut ei: usize,
    units: &[ModuleUnit],
    idx_of: &FxHashMap<&str, usize>,
    state: &mut [u8],
    depth: &mut [usize],
    stack: &mut Vec<(usize, usize)>,
) -> bool {
    let deps = &units[u].dependents;
    while ei < deps.len() {
        let dep_name = deps[ei].as_str();
        ei += 1;
        let Some(&v) = idx_of.get(dep_name) else {
            depth[u] = depth[u].max(1);
            continue;
        };
        match state[v] {
            0 => {
                stack.push((u, ei));
                state[v] = 1;
                stack.push((v, 0));
                return true;
            }
            2 => depth[u] = depth[u].max(1 + depth[v]),
            _ => depth[u] = depth[u].max(1),
        }
    }
    false
}

pub fn units_interfere(a: &ModuleUnit, b: &ModuleUnit) -> bool {
    if a.id == b.id || a.deps.contains(&b.id) || b.deps.contains(&a.id) {
        return true;
    }
    let a_files: HashSet<&str> = a.files.iter().map(String::as_str).collect();
    b.files.iter().any(|f| a_files.contains(f.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::char_prefix;
    use crate::testutil::{TempRepo, asrf_edit, map1, plan_repo, unit_stub};

    fn plan_files(tag: &str, files: &[(&str, &str)]) -> (TempRepo, DagPlan) {
        let dir = TempRepo::with_files(tag, files);
        let plan = plan_repo(dir.path(), &HygieneScorer::new());
        (dir, plan)
    }

    fn run_patch(base: &BTreeMap<String, String>, raw: &str) -> Result<AppliedPatch> {
        apply_patches(base, &parse_patches(raw))
    }

    fn edit_one(rel: &str, orig: &str, search: &str, replace: &str) -> Result<String> {
        let base = map1(rel, orig.to_string());
        let mut out = run_patch(&base, &asrf_edit(rel, search, replace))?;
        Ok(out.updated.remove(rel).unwrap_or_default())
    }

    #[test]
    fn test_packet_slicing_and_wave_context() {
        let files = [
            ("src/a.py", "def alpha(x: int) -> int:\n    return x\n"),
            ("src/b.rs", "pub fn beta(y: i32) -> i32 { y }\n"),
        ];
        let (dir, plan) = plan_files("pkt", &files);

        let unit = unit_stub("a", &["src/a.py"]);
        let p1 = build_packet(dir.path(), &unit, &["z_err".into(), "a_err".into(), "z_err".into()], "", 4000).unwrap();
        let p2 = build_packet(dir.path(), &unit, &["a_err".into(), "z_err".into()], "", 4000).unwrap();
        assert_eq!((&p1, p1.matches("z_err").count()), (&p2, 1));
        assert!(p1.find("a_err").unwrap() < p1.find("z_err").unwrap());

        let mut src = String::from("import os\n\n");
        for i in 0..25 {
            src.push_str(&format!("def clean_{i}(x: int) -> int:\n    return x + {i}\n\n"));
        }
        src.push_str("@app.get('/hot')\ndef hot_target(items=[]):\n    if items:\n        if len(items) > 1:\n            return items[0]\n    return None\n");
        let sliced = packet::slice_anchored(&src, 400, &HashSet::new());
        assert!(sliced.len() < src.len() && sliced.contains("def hot_target(items=[]):"));
        assert!(sliced.contains("@app.get('/hot')"), "decorator detached from hot_target: {sliced}");
        assert!(sliced.contains("// stub: def clean_"));

        let mut utf = String::from("# こんにちは世界 🚀 café résumé\n");
        for i in 0..40 {
            utf.push_str(&format!("def fn_{i}():\n    return '🚀日本語テスト_{i}'\n\n"));
        }
        let utf_slice = packet::slice_anchored(&utf, 120, &HashSet::new());
        assert!(utf_slice.contains("def fn_0") && utf_slice.len() < utf.len());
        assert_eq!(char_prefix("🚀日本語", 2), "🚀日");

        let fwd = wave_context(&plan.units, dir.path());
        let mut rev_u = plan.units.clone();
        rev_u.reverse();
        assert_eq!(fwd, wave_context(&rev_u, dir.path()));
        assert!(fwd.chars().count() <= 1200 && fwd.contains("def alpha") && fwd.contains("pub fn beta"));
    }

    #[test]
    fn test_polyglot_abi_hashing() {
        let diff_cases = [
            ("a.rs", "pub fn calc(x: i32) -> i32 { x }", "pub fn calc_v2(x: i32) -> i32 { x }"),
            ("b.ts", "export function render(id: string) { return id; }", "export function renderCard(id: string) { return id; }"),
            ("c.tsx", "export function Card({ title }: Props) { return title; }\n", "export function Card({ label }: Props) { return label; }\n"),
            ("c.go", "package p\nfunc Build(x int) int { return x }", "package p\nfunc BuildAll(x int) int { return x }"),
            ("svc.go", "package p\nfunc (s *Svc) Handle(id string) error { return nil }\n", "package p\nfunc (s *Svc) HandleV2(id string) error { return nil }\n"),
        ];
        for (rel, a, b) in diff_cases {
            assert_ne!(hash_abi(rel, a), hash_abi(rel, b), "{rel}");
        }

        let rs_flat = "pub fn compute(a: i32, b: i32) -> i32 { a + b }\n";
        let rs_wrap = "pub fn compute(\n    a: i32,\n    b: i32,\n) -> i32 {\n    a + b\n}\n";
        assert_eq!(hash_abi("x.rs", rs_flat), hash_abi("x.rs", rs_wrap));

        let untyped = "def compute(a, b=1):\n    return a + b\n";
        let typed = "def compute(a: int, b: int = 1) -> int:\n    return a + b\n";
        let renamed = "def compute(x: int, b: int = 1) -> int:\n    return x + b\n";
        assert_eq!(python_abi(untyped), python_abi(typed));
        assert_ne!(python_abi(untyped), python_abi(renamed));
        assert_ne!(
            python_abi("def calc(\n    x: int,\n    y: str,\n) -> float:\n    return 1.0\n"),
            python_abi("def calc(\n    x: int,\n    z: str,\n) -> float:\n    return 1.0\n")
        );

        let reg = crate::platform::EpochRegistry::new();
        let v1 = "pub fn compute(x: i32) -> i32 { x + 1 }\n";
        let v2_same = "pub fn compute(x: i32) -> i32 { let y = x * 2; y - x + 1 }\n";
        let v3_changed = "pub fn compute(x: i32, scale: i32) -> i32 { x * scale }\n";
        let c1 = check_interface_cutoff("packages/calc", &[("src/calc.rs", v1)], None, &reg);
        assert!(c1.changed);
        let c2 = check_interface_cutoff(
            "packages/calc",
            &[("src/calc.rs", v2_same)],
            Some(&c1.interface_blake3),
            &reg,
        );
        assert!(!c2.changed && c2.epoch == c1.epoch && c2.interface_blake3 == c1.interface_blake3);
        let c3 = check_interface_cutoff(
            "packages/calc",
            &[("src/calc.rs", v3_changed)],
            Some(&c2.interface_blake3),
            &reg,
        );
        assert!(c3.changed && c3.epoch > c2.epoch && c3.interface_blake3 != c2.interface_blake3);
    }

    #[test]
    fn test_dag_scc_waves_and_impact() {
        let (_, poly) = plan_files("g_poly", &[
            ("src/a.py", "def py_fn(x):\n    return x\n"),
            ("src/b.rs", "pub fn rs_fn(x: i32) -> i32 { x }\n"),
            ("src/c.ts", "export function tsFn(x: number) { return x; }\n"),
            ("src/D.tsx", "export function Card() { return <div/>; }\n"),
            ("src/e.js", "export function jsFn(x) { return x; }\n"),
            ("src/f.go", "package main\nfunc GoFn(x int) int { return x }\n"),
        ]);
        assert_eq!((poly.total_files, poly.units.len()), (6, 6));

        let (_, scc) = plan_files("g_scc", &[
            ("pkg/a.py", "from pkg.b import fb\ndef fa():\n    return fb()\n"),
            ("pkg/b.py", "from pkg.a import fa\ndef fb():\n    return fa()\n"),
        ]);
        assert_eq!((scc.scc_count, scc.units.len(), scc.units[0].files.len()), (1, 1, 2));
        assert!(scc.units[0].allow_abi_change);

        let (_, chain) = plan_files("g_chain", &[
            ("pkg/root_core.py", "def core_v():\n    return 1\n"),
            ("pkg/mid_layer.py", "from pkg.root_core import core_v\ndef mid_v():\n    return core_v()\n"),
            ("pkg/top_api.py", "from pkg.mid_layer import mid_v\ndef top_v():\n    return mid_v()\n"),
            ("pkg/isolated.py", "def iso():\n    return 0\n"),
            ("tests/test_smoke_suite.py", "from pkg.root_core import core_v\ndef test_c():\n    assert core_v() == 1\n"),
        ]);
        let wave_of = |id: &str| {
            let found = chain.units.iter().find(|u| u.id == id);
            found.unwrap().wave
        };
        assert_eq!((wave_of("root_core"), wave_of("mid_layer"), wave_of("top_api")), (0, 1, 2));
        let d = critical_depths(&chain.units);
        assert_eq!((d["root_core"], d["mid_layer"], d["top_api"], d["isolated"]), (2, 1, 0, 0));
        assert_eq!(impacted_tests(&chain, &["pkg/root_core.py".into()]), vec!["tests/test_smoke_suite.py".to_string()]);
        assert!(impacted_tests(&chain, &["pkg/isolated.py".into()]).is_empty());

        let (_, clust) = plan_files("g_clust", &[
            ("alpha/z_mod.rs", "pub fn z() -> i32 { 1 }\n"), ("alpha/a_mod.py", "def a():\n    return 1\n"),
            ("beta/b_mod.py", "def b():\n    return 2\n"), ("alpha/c_mod.rs", "pub fn c() -> i32 { 3 }\n"),
        ]);
        let exts: Vec<&str> = clust.units.iter().map(|u| ext_str(Path::new(&u.files[0]))).collect();
        assert_eq!(exts, vec!["py", "py", "rs", "rs"]);

        let (_, dup_id) = plan_files("g_ids", &[
            ("app/auth/route.ts", "export function GET() { return 1; }\n"),
            ("app/billing/route.ts", "export function GET() { return 2; }\n"),
        ]);
        let ids: HashSet<&str> = dup_id.units.iter().map(|u| u.id.as_str()).collect();
        assert!(ids.len() == 2 && ids.contains("app/auth/route") && ids.contains("app/billing/route"));

        let a = unit_stub("a", &["a.py"]);
        let mut b = unit_stub("b", &["b.py"]);
        b.deps.push("a".into());
        assert!(units_interfere(&a, &b) && !units_interfere(&a, &unit_stub("c", &["c.py"])));
    }

    #[test]
    fn test_asrf_patch_ops_and_security() {
        let crlf = map1("pkg/a.py", "x = 1\r\ny = 2\r\n".to_string());
        let raw = format!("{}\r\n{}", asrf_edit("a.py", "x = 1", "x = 10"), asrf_edit("pkg/a.py", "y = 2", "y = 20"));
        assert_eq!(parse_patches(&raw).len(), 2);
        assert_eq!(run_patch(&crlf, &raw).unwrap().updated["pkg/a.py"], "x = 10\ny = 20\n");

        assert!(patch::replace_window("    val = 1\n    val = 1\n", "val = 1", "val = 2", |a, b| a.trim() == b.trim()).is_none());
        let two = BTreeMap::from([("a.py".into(), "x = 1\n".into()), ("b.py".into(), "y = 2\n".into())]);
        assert!(apply_patches(&two, &[]).is_err() && run_patch(&two, &asrf_edit("ghost.py", "x = 1", "x = 2")).is_err());
        assert!(edit_one("a.py", "x = 1\nx = 1\n", "x = 1", "x = 2").is_err());

        let orig = "pub fn run(x: i32) -> i32 {\n    let step1 = x + 1;\n    let untouched_middle = step1 * 42;\n    untouched_middle\n}\n";
        let s_el = "pub fn run(x: i32) -> i32 {\n// ... existing code ...\n    untouched_middle\n}";
        let r_el = "pub fn run(x: i32) -> i32 {\n    let _checked = x >= 0;\n// ... existing code ...\n    untouched_middle + 1\n}";
        let out = edit_one("src/calc.rs", orig, s_el, r_el).unwrap();
        assert!(out.contains("let untouched_middle = step1 * 42;") && out.contains("let _checked = x >= 0;") && out.contains("untouched_middle + 1"));
        assert!(edit_one("src/dup.py", "def dup():\n    return 1\ndef dup():\n    return 2\n", "def dup():\n# ... existing code ...\n    return 2", "def dup():\n# ... existing code ...\n    return 3").is_err());

        let ops_base = BTreeMap::from([
            ("src/old.ts".into(), "export const a = 1;\n".into()),
            ("src/dead.ts".into(), "export const d = 0;\n".into()),
        ]);
        let ops_patch = concat!(
            "<<<REFIT_PATCH path=\"src/shared.ts\" op=\"create\">>>\n<<<SEARCH>>>\n<<<REPLACE>>>\nexport const shared = 42;\n<<<END>>>\n",
            "<<<REFIT_PATCH path=\"src/dead.ts\" op=\"delete\">>>\n<<<SEARCH>>>\n<<<REPLACE>>>\n<<<END>>>\n",
            "<<<REFIT_PATCH path=\"src/old.ts\" op=\"rename\" new_path=\"src/core/new.ts\">>>\n<<<SEARCH>>>\na = 1\n<<<REPLACE>>>\na = 2\n<<<END>>>"
        );
        let ops = run_patch(&ops_base, ops_patch).unwrap();
        assert!(ops.created.contains(&"src/shared.ts".to_string()) && ops.deleted.contains(&"src/dead.ts".to_string()));
        assert!(ops.updated["src/core/new.ts"].contains("a = 2"));

        let sec_base = map1("src/a.rs", "pub fn keep(a: i32, b: i32) -> i32 { a + b }\n".into());
        for evil in [
            "<<<REFIT_PATCH path=\"../../pwn.rs\" op=\"create\">>>\n<<<REPLACE>>>\npwn\n<<<END>>>",
            "<<<REFIT_PATCH path=\".git/hooks/pre-commit\" op=\"create\">>>\n<<<REPLACE>>>\npwn\n<<<END>>>",
            "<<<REFIT_PATCH op=\"rename\" new_path=\"../escape.rs\" path=\"src/a.rs\">>>\n<<<END>>>",
        ] {
            assert!(run_patch(&sec_base, evil).is_err(), "{evil}");
        }

        let ren = parse_patches("<<<REFIT_PATCH op=\"rename\" new_path=\"src/b.rs\" path=\"src/a.rs\">>>\n<<<END>>>");
        assert_eq!((ren[0].path.as_str(), ren[0].new_path.as_deref()), ("src/a.rs", Some("src/b.rs")));
        let moved = apply_patches(&sec_base, &ren).unwrap();
        assert!(moved.updated.contains_key("src/b.rs") && moved.deleted.contains(&"src/a.rs".to_string()));
    }
}
