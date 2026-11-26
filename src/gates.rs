mod audit;
mod rules;
mod runners;
mod stage;

#[cfg(test)]
pub use audit::{detect_langs, detect_tools};
pub use audit::{
    AuditStats, GateConfig, ScanCache, audit_repo, read_gates, scan_cached, scan_repo,
};
pub use rules::check_rules;
pub use runners::net_new_diagnostics;
#[cfg(test)]
pub use stage::{decode_lsp, encode_lsp, stage_copies};
pub use stage::{prune_repo_caches, stage_hardlinks, sweep_stages};

use crate::graph::hash_abi;
use crate::hygiene::HygieneScorer;
use crate::platform::{env_truthy, par_join2, par_map, run_cmd, sandbox_env};
use crate::types::{FileEgoContext, GateVerdict, ModuleUnit};
use anyhow::Result;
use runners::{
    CmdSpec, RunAccum, check_g0, cmd_args, detect_cold_js_linter, detect_cold_ts_checker,
    first_lines, first_pair, merged_rust_cargo_env, parse_pytest, parse_ruff, parse_ty, poly_test,
    py_test_env, run_fmt, run_rust_check, rust_edition,
};
use stage::{CargoSlot, materialize_stage};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Default)]
struct LangPaths {
    all: Vec<String>, py: Vec<String>, js: Vec<String>,
    rs: Vec<String>, ts: Vec<String>, go: Vec<String>,
}

struct StageCtx<'a> {
    cfg: &'a GateConfig, root: &'a Path, env: &'a [(String, String)],
    paths: &'a LangPaths, slot: Option<&'a CargoSlot>,
}

impl StageCtx<'_> {
    fn cmd<'b>(&'b self, prog: &'b str, args: &'b [String], ms: u64) -> CmdSpec<'b> {
        CmdSpec::new(prog, args, self.root, self.env, ms)
    }
}

fn group_paths(files: &BTreeMap<String, String>) -> LangPaths {
    let mut lp = LangPaths::default();
    for rel in files.keys() {
        lp.all.push(rel.clone());
        let bucket = match Path::new(rel).extension().and_then(|e| e.to_str()) {
            Some("py") => &mut lp.py,
            Some("js" | "mjs" | "cjs") => &mut lp.js,
            Some("rs") => &mut lp.rs,
            Some("ts" | "tsx") => &mut lp.ts,
            Some("go") => &mut lp.go,
            _ => continue,
        };
        bucket.push(rel.clone());
    }
    lp
}

pub fn run_gates(
    cfg: &GateConfig, unit: &ModuleUnit, candidates: &BTreeMap<String, String>,
    deleted: &[String], hygiene: &HygieneScorer,
) -> Result<GateVerdict> {
    stage::ensure_cache_dirs_tagged(&cfg.repo_root);
    let stage = materialize_stage(&cfg.repo_root, candidates, deleted)?;
    let base_env = sandbox_env(&stage::polyglot_cache_env(&cfg.repo_root));
    let paths = group_paths(candidates);
    let cargo_slot = (!paths.rs.is_empty() && stage.join("Cargo.toml").exists())
        .then(|| CargoSlot::acquire(&cfg.repo_root));
    let ctx = StageCtx { cfg, root: &stage, env: &base_env, paths: &paths, slot: cargo_slot.as_ref() };

    let mut verdict = GateVerdict {
        gate0_abi_ok: true, gate2_score_before: unit.baseline_hygiene,
        gate2_score_after: unit.baseline_hygiene, deleted_files: deleted.to_vec(), ..GateVerdict::default()
    };

    let g0_t = Instant::now();
    if let Some((gate, msg)) = run_gate0(&ctx, candidates) {
        verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
        verdict.failed_gate = Some(gate);
        verdict.diagnostics.push(msg);
        return Ok(verdict);
    }

    verdict.gate0_syntax_ok = true;
    if !check_abi(unit, candidates, "", &mut verdict) {
        verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
        return Ok(verdict);
    }
    verdict.gate0_ms = g0_t.elapsed().as_millis() as u64;
    if !check_invariants(&cfg.repo_root, candidates, &mut verdict) { return Ok(verdict); }

    let g3_start = Instant::now();
    verdict.formatted_files = run_formatters(&ctx);
    let mut acc = RunAccum::default();
    run_ruff(&ctx, &mut verdict, &mut acc);

    let mut working = BTreeMap::new();
    for rel in &paths.all { working.insert(rel.clone(), fs::read_to_string(ctx.root.join(rel))?); }
    if !check_abi(unit, &working, " after formatter/autofix", &mut verdict)
        || !check_invariants(&cfg.repo_root, &working, &mut verdict)
    {
        return Ok(verdict);
    }
    if !paths.py.is_empty() { run_py(&ctx, unit, &mut verdict, &mut acc); }
    run_poly(&ctx, unit, &mut verdict, &mut acc);
    verdict.gate3_ms = g3_start.elapsed().as_millis() as u64;
    if !acc.ran_tests { verdict.gate1_skipped_reason = Some("no_matching_tests_or_manifest_found".into()); }
    let (hints, strict_veto) = score_gate2(cfg, unit, &working, hygiene, &mut verdict);
    finalize_verdict(cfg, working, acc, hints, strict_veto, &mut verdict);
    Ok(verdict)
}

fn check_py0(ctx: &StageCtx<'_>) -> Option<(String, String)> {
    let script = "import ast, sys\nfor p in sys.argv[1:]:\n    \
                  ast.parse(open(p, encoding='utf-8').read(), filename=p)";
    let args = cmd_args(&["-c", script], &ctx.paths.py);
    match run_cmd(&ctx.cfg.python_bin, &args, ctx.root, ctx.env, true, 5_000) {
        Ok(o) if o.exit_code == 0 && !o.timed_out => None,
        Ok(o) => Some(("gate0_syntax".into(), format!("Gate 0 syntax error: {}", first_lines(&o.stderr, 4)))),
        Err(e) => Some(("gate0_syntax".into(), format!("Gate 0 python spawn error: {e}"))),
    }
}

fn check_js0(ctx: &StageCtx<'_>, strict: bool) -> Option<(String, String)> {
    par_map(&ctx.paths.js, |js| {
        let args = cmd_args(&["--check", js], &[]);
        check_g0("gate0_syntax", &format!("node --check for `{js}`"), &ctx.cmd("node", &args, 5_000), strict, false)
    })
    .into_iter().flatten().next()
}

fn run_gate0(ctx: &StageCtx<'_>, candidates: &BTreeMap<String, String>) -> Option<(String, String)> {
    for (rel, content) in candidates {
        if let Err(err) = crate::lex::check_balanced_syntax(rel, content) {
            return Some(("gate0_syntax".into(), format!("Gate 0 polyglot syntax error in `{rel}`: {err}")));
        }
    }

    let strict = ctx.cfg.strict_polyglot_tools;
    let active: Vec<&str> = [
        (!ctx.paths.py.is_empty()).then_some("py"),
        (!ctx.paths.js.is_empty()).then_some("js"),
        ctx.slot.is_some().then_some("rs"),
        (!ctx.paths.ts.is_empty() && ctx.root.join("tsconfig.json").exists()).then_some("ts"),
        (!ctx.paths.go.is_empty() && ctx.root.join("go.mod").exists()).then_some("go"),
    ]
    .into_iter().flatten().collect();

    par_map(&active, |&lang| match lang {
        "py" => check_py0(ctx),
        "js" => check_js0(ctx, strict),
        "rs" => ctx.slot.and_then(|s| run_rust_check(&ctx.cfg.repo_root, ctx.root, s, true, strict)),
        "ts" => {
            let (prog, args, label) = detect_cold_ts_checker();
            check_g0("gate0_ts_check", label, &ctx.cmd(prog, &args, 20_000), strict, true)
        }
        _ => check_g0("gate0_go_vet", "go vet", &ctx.cmd("go", &cmd_args(&["vet", "./..."], &[]), 20_000), strict, false),
    })
    .into_iter().flatten().next()
}

fn check_abi(unit: &ModuleUnit, files: &BTreeMap<String, String>, suffix: &str, v: &mut GateVerdict) -> bool {
    if unit.allow_abi_change { return true; }
    for (rel, content) in files {
        if unit.abi_hash.get(rel).is_some_and(|exp| &hash_abi(rel, content) != exp) {
            v.gate0_abi_ok = false;
            v.failed_gate = Some("gate0_abi".into());
            v.diagnostics.push(format!("Gate 0 AbiPinViolation{suffix} in `{rel}`: exported signature names changed"));
            return false;
        }
    }
    true
}

fn check_invariants(repo_root: &Path, files: &BTreeMap<String, String>, v: &mut GateVerdict) -> bool {
    let violations = check_rules(repo_root, files);
    if violations.is_empty() { return true; }
    v.failed_gate = Some("gate3_5_invariants".into());
    v.diagnostics.extend(violations.into_iter().take(8));
    false
}

fn run_formatters(ctx: &StageCtx<'_>) -> usize {
    let mut n = run_fmt(&ctx.cfg.ruff_bin, &["format", "--quiet"], &ctx.paths.py, ctx.root, ctx.env);
    if !ctx.paths.rs.is_empty() {
        let ed = rust_edition(ctx.root);
        n += run_fmt("rustfmt", &["--edition", &ed], &ctx.paths.rs, ctx.root, ctx.env);
    }
    n += run_fmt("gofmt", &["-w"], &ctx.paths.go, ctx.root, ctx.env);
    if !ctx.paths.ts.is_empty() || !ctx.paths.js.is_empty() {
        let mut both = ctx.paths.ts.clone();
        both.extend(ctx.paths.js.iter().cloned());
        n += run_fmt("biome", &["format", "--write"], &both, ctx.root, ctx.env);
    }
    n
}

fn run_ruff(ctx: &StageCtx<'_>, verdict: &mut GateVerdict, acc: &mut RunAccum) {
    if ctx.paths.py.is_empty() { return; }
    let args = cmd_args(&["check", "--fix", "--output-format=json"], &ctx.paths.py);
    match run_cmd(&ctx.cfg.ruff_bin, &args, ctx.root, ctx.env, true, 8_000) {
        Ok(out) => {
            acc.ruff_diags = parse_ruff(&out.stdout);
            verdict.gate3_ruff_errors = acc.ruff_diags.len();
        }
        Err(e) if ctx.cfg.strict_polyglot_tools => {
            acc.poly_lint.push(format!("ruff unavailable (strict mode): {e}"));
        }
        Err(_) => {}
    }
}

fn run_py(ctx: &StageCtx<'_>, unit: &ModuleUnit, verdict: &mut GateVerdict, acc: &mut RunAccum) {
    let has_tests = !unit.test_files.is_empty() || ctx.root.join("tests").exists();
    let run_ty = || {
        let args = cmd_args(&["check", "--error-on-warning"], &ctx.paths.py);
        run_cmd(&ctx.cfg.ty_bin, &args, ctx.root, ctx.env, true, 10_000)
    };

    let run_pt = || {
        let t = Instant::now();
        let args = if unit.test_files.is_empty() { cmd_args(&["-q", "--tb=short", "tests"], &[]) }
            else { cmd_args(&["-q", "--tb=short"], &unit.test_files) };
        let env = py_test_env(&ctx.cfg.repo_root, ctx.root);
        let out = run_cmd(&ctx.cfg.pytest_bin, &args, ctx.root, &env, true, 15_000);
        (out, t.elapsed().as_millis() as u64)
    };

    let (ty_res, (pt_res, g1_ms)) = par_join2(run_ty, || {
        has_tests.then(run_pt).map_or((None, 0), |(r, m)| (Some(r), m))
    });
    apply_py_res(ctx.cfg.strict_polyglot_tools, ty_res, pt_res, g1_ms, verdict, acc);
}

fn apply_py_res(
    strict: bool, ty_res: Result<crate::types::SandboxResult>,
    pt_res: Option<Result<crate::types::SandboxResult>>, g1_ms: u64,
    verdict: &mut GateVerdict, acc: &mut RunAccum,
) {
    apply_ty_res(strict, ty_res, verdict, acc);
    if let Some(pt_r) = pt_res { apply_pt_res(strict, pt_r, g1_ms, verdict, acc); }
}

fn apply_ty_res(
    strict: bool, ty_res: Result<crate::types::SandboxResult>, verdict: &mut GateVerdict, acc: &mut RunAccum,
) {
    match ty_res {
        Ok(o) if o.exit_code == 0 => {}
        Ok(o) => {
            acc.ty_diags = parse_ty(&o.stdout, &o.stderr);
            verdict.gate3_ty_errors = acc.ty_diags.len();
        }
        Err(e) if strict => {
            acc.ty_diags.push(format!("ty unavailable (strict mode): {e}"));
            verdict.gate3_ty_errors = 1;
        }
        Err(_) => {}
    }
}

fn apply_pt_res(
    strict: bool, pt_r: Result<crate::types::SandboxResult>, g1_ms: u64,
    verdict: &mut GateVerdict, acc: &mut RunAccum,
) {
    let out = match pt_r {
        Ok(o) => o,
        Err(e) if strict => {
            verdict.gate1_tests_failed += 1;
            acc.pt_exit = 1;
            acc.pt_stdout.push_str(&format!("pytest unavailable (strict mode): {e}"));
            return;
        }
        Err(_) => return,
    };

    acc.ran_tests = true;
    let (passed, failed) = parse_pytest(&out.stdout, &out.stderr, out.exit_code);
    verdict.gate1_tests_passed += passed;
    verdict.gate1_tests_failed += failed;
    verdict.gate1_ms += g1_ms;
    verdict.gate1_cpu_ms += out.cpu_ms;
    verdict.gate1_peak_rss_bytes = verdict.gate1_peak_rss_bytes.max(out.peak_rss_bytes);
    (acc.pt_exit, acc.pt_stdout) = (out.exit_code, out.stdout);
}

fn run_rust_poly(ctx: &StageCtx<'_>, slot: &CargoSlot, strict: bool, verdict: &mut GateVerdict, acc: &mut RunAccum) {
    let env = merged_rust_cargo_env(&slot.dir, slot.rust_cache_env(&ctx.cfg.repo_root, true));
    let t_args = cmd_args(&["test", "--quiet"], &[]);
    poly_test("cargo test", &CmdSpec::new("cargo", &t_args, ctx.root, &env, 45_000), strict, 8, verdict, acc);
    let c_args = cmd_args(&["clippy", "--no-deps", "--quiet", "--", "-D", "warnings"], &[]);
    if let Ok(o) = run_cmd("cargo", &c_args, ctx.root, &env, true, 30_000) && o.exit_code != 0 && !o.timed_out {
        let msg = first_pair(&o.stdout, &o.stderr, 4);
        if !msg.contains("no such command") { acc.poly_lint.push(format!("cargo clippy: {msg}")); }
    }
}

fn run_poly(ctx: &StageCtx<'_>, unit: &ModuleUnit, verdict: &mut GateVerdict, acc: &mut RunAccum) {
    let strict = ctx.cfg.strict_polyglot_tools;
    if let Some(slot) = ctx.slot { run_rust_poly(ctx, slot, strict, verdict, acc); }
    if !ctx.paths.go.is_empty() && ctx.root.join("go.mod").exists() {
        poly_test("go test", &ctx.cmd("go", &cmd_args(&["test", "./..."], &[]), 30_000), strict, 6, verdict, acc);
    }
    let has_js_ts = !ctx.paths.ts.is_empty() || !ctx.paths.js.is_empty();
    if has_js_ts && !unit.test_files.is_empty() {
        run_js_tests(ctx, &unit.test_files, strict, verdict, acc);
    }

    if !has_js_ts { return; }
    let (prog, mut args, label) = detect_cold_js_linter(ctx.root.join("tsconfig.json").exists());
    args.extend(ctx.paths.ts.iter().chain(&ctx.paths.js).cloned());
    match run_cmd(prog, &args, ctx.root, ctx.env, true, 10_000) {
        Ok(o) if o.exit_code != 0 && !o.timed_out => {
            acc.poly_lint.push(format!("{label}: {}", first_pair(&o.stdout, &o.stderr, 4)));
        }
        Err(e) if strict => acc.poly_lint.push(format!("{prog} unavailable (strict mode): {e}")),
        _ => {}
    }
}

fn run_js_tests(
    ctx: &StageCtx<'_>, test_files: &[String], strict: bool, verdict: &mut GateVerdict, acc: &mut RunAccum,
) {
    let (mut js_t, mut ts_t) = (Vec::new(), Vec::new());
    for t in test_files {
        if t.ends_with(".js") || t.ends_with(".mjs") || t.ends_with(".cjs") { js_t.push(t.clone()); }
        else if t.ends_with(".ts") || t.ends_with(".tsx") { ts_t.push(t.clone()); }
    }
    if !js_t.is_empty() {
        poly_test("node --test", &ctx.cmd("node", &cmd_args(&["--test"], &js_t), 15_000), strict, 6, verdict, acc);
    }
    if !ts_t.is_empty() {
        poly_test("vitest", &ctx.cmd("vitest", &cmd_args(&["run"], &ts_t), 25_000), false, 6, verdict, acc);
    }
}

fn record_strict_vetoes(cfg: &GateConfig, rel: &str, txt: &str, vetoes: &[String], hints: &mut Vec<String>) {
    let before = fs::read_to_string(cfg.repo_root.join(rel)).unwrap_or_default();
    let a = crate::hygiene::strict::audit_anti_gaming(rel, &before, txt);
    if !a.passed {
        hints.push(format!(
            "codehygiene-strict `{}` veto (cap {:.2}, indent {}:{}, before={} after={} new={})",
            a.path, a.veto_cap(), a.indent_mode, a.indent_quantum,
            a.before_vetoes.len(), a.after_vetoes.len(), a.introduced_vetoes.len(),
        ));
    }
    for v in vetoes.iter().take(4) { hints.push(format!("codehygiene-strict `{rel}`: {v}")); }
}

fn score_gate2(
    cfg: &GateConfig, unit: &ModuleUnit, working: &BTreeMap<String, String>,
    hygiene: &HygieneScorer, verdict: &mut GateVerdict,
) -> (Vec<String>, bool) {
    let (t, strict) = (Instant::now(), env_truthy("REFIT_STRICT_HYGIENE"));
    let (mut sum, mut hints, mut has_veto) = (0.0f64, Vec::new(), false);
    let deg = unit.dependents.len() + unit.deps.len();
    let is_tangle = unit.dependents.len() >= 6 && unit.deps.len() >= 6 && deg > 0
        && (0.35..=0.65).contains(&((unit.deps.len() as f64) / (deg as f64)));
    let unit_ego = FileEgoContext {
        fan_in: unit.dependents.len(), fan_out: unit.deps.len(), neighbor_dirs: 2,
        has_module_cycle: unit.files.len() > 1, tangle: is_tangle, ..FileEgoContext::default()
    };

    for (rel, txt) in working {
        let h = if strict { hygiene.score_strict_ego(rel, txt, Some(&unit_ego)) }
            else { hygiene.score_content(rel, txt) };
        sum += if strict { h.multiscale_score.or(h.strict_score).unwrap_or(h.score) } else { h.score };
        if strict && !h.triggered_vetoes.is_empty() {
            has_veto = true;
            record_strict_vetoes(cfg, rel, txt, &h.triggered_vetoes, &mut hints);
        }
        hints.extend(h.squiggles.iter().take(6).map(|sq| {
            format!("codehygiene `{rel}` L{}: {} ({:.2}) {}", sq.line, sq.kind, sq.probability, sq.message)
        }));
    }
    verdict.gate2_score_after = if working.is_empty() {
        cfg.min_hygiene_target.max(unit.baseline_hygiene)
    } else {
        sum / (working.len() as f64)
    };
    verdict.gate2_ms = t.elapsed().as_millis() as u64;
    (hints, has_veto)
}

fn finalize_verdict(
    cfg: &GateConfig, working: BTreeMap<String, String>, acc: RunAccum,
    hints: Vec<String>, strict_veto: bool, v: &mut GateVerdict,
) {
    if acc.pt_exit != 0 || v.gate1_tests_failed > 0 {
        let preview = first_lines(&acc.pt_stdout, 8);
        v.failed_gate = Some("gate1_tests".into());
        v.diagnostics.push(format!("Gate 1 tests failed ({} failed): {preview}", v.gate1_tests_failed));
    }

    if v.gate3_ruff_errors > 0 || !acc.poly_lint.is_empty() {
        v.failed_gate.get_or_insert_with(|| "gate3_ruff".into());
        v.diagnostics.extend(acc.ruff_diags.into_iter().take(6));
        v.diagnostics.extend(acc.poly_lint.into_iter().take(4));
    }

    if v.gate3_ty_errors > 0 {
        v.failed_gate.get_or_insert_with(|| "gate3_ty".into());
        v.diagnostics.extend(acc.ty_diags.into_iter().take(6));
    }

    let improved = !strict_veto && v.gate2_score_after >= v.gate2_score_before + 0.12 && v.gate2_score_after >= 0.72;
    if strict_veto || (v.gate2_score_after < cfg.min_hygiene_target && !improved) {
        v.failed_gate.get_or_insert_with(|| "gate2_hygiene".into());
        v.diagnostics.push(format!(
            "Gate 2 codehygiene too low: {:.3} (before {:.3}, target >= {:.2})",
            v.gate2_score_after, v.gate2_score_before, cfg.min_hygiene_target
        ));
        v.diagnostics.extend(hints.into_iter().take(6));
    }

    if v.failed_gate.is_none() {
        v.passed = true;
        for (rel, txt) in working { v.final_contents.insert(rel, txt.into_bytes()); }
    }
}

#[cfg(test)]
fn verify_stage_gc_and_delete(tmp: &crate::testutil::TempRepo) {
    let (g1, g2) = (CargoSlot::acquire(tmp.path()), CargoSlot::acquire(tmp.path()));
    assert!(g1.slot != g2.slot && !g1.dir.starts_with(tmp.path()));

    let parent = tmp.path().parent().unwrap();
    let stale = parent.join(".refit-stages/tx-orphan-old");
    fs::create_dir_all(&stale).unwrap();
    fs::write(stale.join("junk.txt"), "stale").unwrap();
    let _ = stage_copies();
    assert!(sweep_stages(tmp.path(), 0) >= 1 && !stale.exists() && g1.dir.exists());
    drop((g1, g2));

    let mut unit = crate::testutil::unit_stub("del_dead", &["dead.rs"]);
    unit.allow_abi_change = true;
    let del = ["dead.rs".to_string()];
    let v = run_gates(&read_gates(tmp.path()), &unit, &BTreeMap::new(), &del, &HygieneScorer::new()).unwrap();
    assert!(v.passed && v.gate2_score_after >= 0.85, "{:?}", v.diagnostics);
}

#[cfg(test)]
fn verify_gate35_rules(root: &Path) {
    let aws = ["AKIA", "IOSFODNN7EXAMPLE"].concat();
    let cand = BTreeMap::from([
        ("src/core/ok.rs".into(), "// never call .lock().unwrap()\npub fn ok() -> i32 { 1 }\n".into()),
        ("src/ui/page.ts".into(), "import { query } from '../src/db/pool';\nexport const p = query;\n".into()),
        ("src/cfg.py".into(), format!("KEY = \"{aws}\"\n")),
    ]);
    let v = check_rules(root, &cand);
    assert!(!v.iter().any(|m| m.contains("src/core/ok.rs")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("ui_no_db") && m.contains("src/ui/page.ts")));
    assert!(v.iter().any(|m| m.contains("aws_access_key_id")));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{TempRepo, map1, plan_repo};

    struct GateCase { _tmp: TempRepo, cfg: GateConfig, unit: ModuleUnit, scorer: HygieneScorer }

    impl GateCase {
        fn new(tag: &str, files: &[(&str, &str)]) -> Self {
            let tmp = TempRepo::with_files(tag, files);
            let (scorer, cfg) = (HygieneScorer::new(), read_gates(tmp.path()));
            let plan = plan_repo(tmp.path(), &scorer);
            let unit = plan.units.into_iter().next().unwrap();
            Self { _tmp: tmp, cfg, unit, scorer }
        }
        fn verify(&self, rel: &str, body: &str) -> GateVerdict {
            run_gates(&self.cfg, &self.unit, &map1(rel, body.into()), &[], &self.scorer).unwrap()
        }
    }

    #[test]
    fn polyglot_brace_lexer() {
        let rs_nested = "/* outer /* nested { */ still comment */\npub fn chars() -> (char, char) { ('é', '🚀') }\n";
        for (rel, body, ok) in [
            ("src/lib.rs", "pub fn f<'a>(s: &'a str) -> char { let _ = \"}\"; '{' }\n", true),
            ("src/lib.rs", "pub fn broken() { let x = 1;\n", false),
            ("src/raw.rs", "pub fn f() -> &'static str { let _b = b'{'; r#\"unbalanced }\"# }\n", true),
            ("src/u.rs", rs_nested, true),
            ("src/app.tsx", "export const C = () => <div>{\"}\"}</div>;\n", true),
            ("src/App.tsx", "export const C = () => <div>Don't panic {1 + 2}</div>;\n", true),
            ("src/b.tsx", "export function Banner() {\n  return <p>\n    Don't panic\n  </p>;\n}\n", true),
            ("src/app.ts", "export function bad(x: number { return x; }\n", false),
            ("src/re.js", "export const r = /[{(]/g;\n", true),
            ("src/r.ts", "export function rx() {\n  return /{/g;\n}\n", true),
            ("src/tpl.ts", "export const s = (x: number) => `val=${foo(\"}\") + x}`;\n", true),
            ("pkg/svc.go", "package p\nfunc Ok() string { return \"{\" }\n", true),
            ("pkg/svc.go", "package p\nfunc Bad() { /* unclosed\n", false),
            ("pkg/raw.go", "package p\nfunc Raw() string { return `unbalanced {\\` }\n", true),
        ] {
            assert_eq!(crate::lex::check_balanced_syntax(rel, body).is_ok(), ok, "case {rel}");
        }
    }

    #[test]
    fn polyglot_audit_and_gate0() {
        let tmp = TempRepo::with_files("poly_audit", &[
            ("src/lib.rs", "pub fn calc(x: i32) -> i32 {\n    x * 2\n}\n"),
            ("src/util.js", "export function calc(x) {\n  return x * 2;\n}\n"),
            ("src/Card.tsx", "export function Card({ title }: { title: string }) {\n  return <section>{title}</section>;\n}\n"),
        ]);
        let langs = detect_langs(tmp.path());
        for l in ["rust", "javascript", "typescript"] { assert!(langs.contains(&l.to_string())); }
        assert_eq!(detect_tools(&langs).len(), 3);

        let rels = vec!["src/lib.rs".into(), "src/util.js".into(), "src/Card.tsx".into()];
        let audit = audit_repo(&read_gates(tmp.path()), &rels, &HygieneScorer::new()).unwrap();
        assert_eq!((audit.tests_failed, audit.ruff_errors, audit.ty_errors), (0, 0, 0));
        assert!(audit.hygiene_median >= 0.85, "got {}", audit.hygiene_median);

        let ts_hdr = "export function doubleVal(x: number): number";
        let gc = GateCase::new("poly_g0", &[("src/math.ts", &format!("{ts_hdr} {{\n  return x * 2;\n}}\n"))]);
        let bad = gc.verify("src/math.ts", &format!("{ts_hdr} {{\n  return x * 2;\n"));
        let good = gc.verify("src/math.ts", &format!("{ts_hdr} {{\n  const out = x * 2;\n  return out;\n}}\n"));
        assert!(!bad.passed && bad.failed_gate.as_deref() == Some("gate0_syntax"));
        assert!(good.passed && good.gate0_syntax_ok);
    }

    #[test]
    fn stage_isolation_slots_and_gc() {
        let files = [
            ("src/keep.py", "VALUE = 1\n"),
            ("src/edit.py", "ORIG = 1\n"),
            ("dead.rs", "pub fn unused() {}\n"),
        ];
        let tmp = TempRepo::nested_with("stage_iso", "target_codebase", &files);
        let (before, cand) = (stage_hardlinks(), map1("src/edit.py", "CAND = 2\n".into()));

        let guard = materialize_stage(tmp.path(), &cand, &[]).unwrap();
        assert!(!guard.starts_with(tmp.path()) && stage_hardlinks() > before);
        assert!(!tmp.has(".refit-stages"));
        fs::write(guard.join("src/edit.py"), "MUTATED_IN_STAGE = 3\n").unwrap();
        assert_eq!(tmp.read("src/edit.py"), "ORIG = 1\n");
        let stage_path = guard.to_path_buf();
        drop(guard);
        assert!(!stage_path.exists());

        verify_stage_gc_and_delete(&tmp);
    }

    #[test]
    fn scan_cache_and_lsp_framing() {
        let framed = encode_lsp(&serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "initialize"}));
        let (decoded, consumed) = decode_lsp(&framed).unwrap();
        assert_eq!((consumed, decoded["method"].as_str()), (framed.len(), Some("initialize")));

        let body = "pub fn a() -> i32 { 1 }\n";
        let tmp = TempRepo::with_files("scancache", &[("src/a.rs", body), ("src/b.rs", "pub fn b() -> i32 { 2 }\n")]);
        let scorer = HygieneScorer::open_persistent(tmp.path());
        let scan = |fast: bool| scan_cached(tmp.path(), true, fast, &scorer).unwrap();

        let (f1, fhit1) = scan(true);
        assert!(!fhit1 && f1.fast_mode && f1.per_file_hashes.len() == 2);
        assert!(scan(true).1);

        let (b1, hit1) = scan(false);
        let (b2, hit2) = scan(false);
        assert!(!hit1 && hit2 && b1.repo_fingerprint == b2.repo_fingerprint);

        std::thread::sleep(std::time::Duration::from_millis(20));
        tmp.write("src/a.rs", body);
        assert!(scan(false).1);

        tmp.write("src/a.rs", "pub fn a() -> i32 { 2 }\n");
        let (b3, hit3) = scan(false);
        assert!(!hit3 && b1.repo_fingerprint != b3.repo_fingerprint);
    }

    fn check_inv_case(gc: &GateCase, body: &str, passed: bool, gate: Option<&str>, needle: Option<&str>) {
        let v = gc.verify("src/lock.rs", body);
        assert_eq!((v.passed, v.failed_gate.as_deref()), (passed, gate));
        if let Some(n) = needle {
            assert!(v.diagnostics.iter().any(|d| d.contains(n)), "{:?}", v.diagnostics);
        }
    }

    #[test]
    fn gate35_invariants_and_secrets() {
        let inv = concat!(
            "rules:\n- id: no_std_mutex_unwrap\n  languages: [rs]\n",
            "  forbid: \".lock().unwrap()\"\n  message: \"use parking_lot::Mutex\"\n",
            "- id: ui_no_db\n  paths: [src/ui/]\n  forbid_import: \"src/db\"\n  message: \"UI cannot import DB\"\n",
        );
        let wrap = |inner: &str| format!("pub fn read_val() -> i32 {{\n    {inner}\n}}\n");
        let gc = GateCase::new("inv35", &[(".refit/invariants.yml", inv), ("src/lock.rs", &wrap("1"))]);

        let fake = ["csk", "-", "1234567890abcdef"].concat();
        check_inv_case(&gc, &wrap("let _g = m.lock().unwrap();\n    1"), false, Some("gate3_5_invariants"), Some("no_std_mutex_unwrap"));
        check_inv_case(&gc, &wrap(&format!("let _k = \"{fake}\";\n    1")), false, Some("gate3_5_invariants"), Some("builtin_no_secrets"));
        check_inv_case(&gc, &wrap("let out = 2;\n    out"), true, None, None);

        verify_gate35_rules(gc._tmp.path());
    }
}
