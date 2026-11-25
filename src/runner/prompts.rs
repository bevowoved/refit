use crate::source::{ext_str, is_js};
use crate::types::ModuleUnit;
use std::collections::BTreeMap;
use std::{fs, path::Path};

const DEFAULT_PROMPT: &str = concat!(
    "You are refit, a surgical polyglot untangling engineer. Emit raw ",
    "<<<REFIT_PATCH path=\"...\">>> <<<SEARCH>>> ... <<<REPLACE>>> ... ",
    "<<<END>>> blocks preserving public exported signatures while passing ",
    "all local verification gates and exact codehygiene v0.7.0 >= 0.85."
);

const PROFILES: [(&str, &str); 4] = [
    ("py", concat!(
        "[Language Profile: Python]\n",
        "- Tooling: ruff format + ruff check + ty --error-on-warning + pytest.\n",
        "- Use explicit PEP 484 annotations, Optional narrowing, and `None` defaults."
    )),
    ("rs", concat!(
        "[Language Profile: Rust]\n",
        "- Tooling: rustfmt + cargo check + cargo test.\n",
        "- Prefer early `?` propagation and small helpers; avoid `.unwrap()` in prod."
    )),
    ("ts", concat!(
        "[Language Profile: TypeScript/JavaScript]\n",
        "- Tooling: biome + tsc --noEmit + node --test.\n",
        "- Prefer strict types, narrow discriminated unions, and clean ESM imports."
    )),
    ("go", concat!(
        "[Language Profile: Go]\n",
        "- Tooling: gofmt + go vet + go test ./...\n",
        "- Always check returned errors explicitly (`if err != nil`); never discard."
    )),
];

pub(super) fn read_prompt(skill_path: &Path) -> String {
    let raw = fs::read_to_string(skill_path).ok();
    raw.filter(|s| s.len() >= 200).unwrap_or_else(|| DEFAULT_PROMPT.into())
}

pub(super) fn prefix_map(base_skill: &str) -> BTreeMap<String, String> {
    PROFILES.into_iter().map(|(l, p)| (l.into(), format!("{base_skill}\n\n{p}"))).collect()
}

pub(super) fn guess_lang(unit: &ModuleUnit) -> &'static str {
    let ext = unit.files.first().map_or("", |f| ext_str(Path::new(f)));
    match ext.to_ascii_lowercase().as_str() {
        "py" => "py",
        "rs" => "rs",
        "go" => "go",
        _ if is_js(ext) => "ts",
        _ => "polyglot",
    }
}

pub(super) fn cold_verifier_hint(lang: &str) -> &'static str {
    match lang {
        "py" => "ruff-cli+ty",
        "rs" => "cargo-check/rust-glancer",
        "ts" => "oxlint/tsrs->biome/tsc",
        "go" => "gofmt+go-vet",
        _ => "polyglot-gates",
    }
}

pub(super) fn pick_effort(unit: &ModuleUnit, attempt: u32, diags: &[String]) -> &'static str {
    let hard = diags.iter().any(|d| {
        let l = d.to_ascii_lowercase();
        l.contains("gate 1") || l.contains("pytest") || l.contains("failed") || l.contains("assert")
    });
    if unit.files.len() > 1 || hard || attempt >= 2 { "medium" } else { "low" }
}
