mod fixture;
mod mutate;
mod verify;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "xtask", about = "Pure-Rust verification, mutation, and fixture tasks for refit")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run S1-S14 cross-platform verification suite.
    Verify(verify::VerifyArgs),
    /// Run T14 8-mutant automated test sensitivity suite.
    Mutate {
        #[arg(long)]
        repo: Option<PathBuf>,
        #[arg(long)]
        out_json: Option<PathBuf>,
    },
    /// Generate deterministic 64-module 10K-LOC Python fixture repo.
    GenFixture {
        #[arg(default_value = "fixture/target_repo")]
        output_dir: PathBuf,
    },
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().cmd {
        Cmd::Verify(args) => verify::run(&root, args),
        Cmd::Mutate { repo, out_json } => {
            mutate::run(repo.as_deref().unwrap_or(&root), out_json.as_deref())
        }
        Cmd::GenFixture { output_dir } => {
            let abs = if output_dir.is_absolute() { output_dir } else { root.join(output_dir) };
            fixture::generate(&abs)
        }
    }
}
