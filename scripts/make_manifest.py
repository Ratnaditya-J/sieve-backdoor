"""Generate a durable manifest for an already-trained artifacts dir.

Use to retro-fit a manifest onto a run whose run_grid predated manifest support
(e.g. pulled back from a remote host). Inventories every adapter so a new
detector can be re-scored offline on the exact population without retraining.

    python scripts/make_manifest.py --model Qwen/Qwen2.5-7B-Instruct \
        --artifacts artifacts/qwen7b --out results/qwen7b
"""
from __future__ import annotations

import argparse

from sieve_backdoors.attacker.common_attacker import git_commit, prereg_content_hash
from sieve_backdoors.grid.persist import build_manifest, save_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = build_manifest(args.artifacts, args.model, config={"retrofit": True},
                       prereg_hash=prereg_content_hash(), git_commit=git_commit())
    p = save_manifest(m, args.out)
    print(f"wrote {p} - {m['n_adapters']} adapters inventoried")


if __name__ == "__main__":
    main()
