#!/usr/bin/env python3
"""Before/after gate table for the fix loop, and a diff.md with the gate changes and the git diff between refs.

usage: scripts/compare_runs.py BEFORE_DIR AFTER_DIR [--before-ref REF] [--after-ref REF] [--repo PATH]
                               [--out PATH] [--rescored]

BEFORE_DIR and AFTER_DIR are benchmark output folders (each with gates.json). Prints the gate table and writes
--out (default AFTER_DIR/../diff.md).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TIER_ORDER = {"photo": 0, "video": 1, "lidar": 2}
RESCORED_NOTE = ("The harness or gates differ between the refs, so both runs were scored with the after ref's "
                 "harness and gates; only the pipeline differs between the two columns.")


def load(run_dir: Path) -> dict:
    path = run_dir / "gates.json"
    if not path.is_file():
        sys.exit(f"compare_runs: {path} not found")
    return json.loads(path.read_text())


def change(b: dict | None, a: dict | None) -> str:
    sb = b["status"] if b else "absent"
    sa = a["status"] if a else "absent"
    if sb == "fail" and sa == "pass":
        return "fixed"
    if sb == "pass" and sa == "fail":
        return "regressed"
    if sb == sa == "fail":
        db, da = b.get("score") or 0.0, a.get("score") or 0.0
        if da < db - 1e-9:
            return "improved"
        return "worse" if da > db + 1e-9 else "unchanged"
    return "unchanged" if sb == sa else f"{sb} -> {sa}"


def rows(before: dict, after: dict) -> list[dict]:
    b = {(r["tier"], r["gate"]): r for r in before.get("rows", [])}
    a = {(r["tier"], r["gate"]): r for r in after.get("rows", [])}
    order = list(b) + [k for k in a if k not in b]
    order.sort(key=lambda k: TIER_ORDER.get(k[0], 9))
    out = []
    for key in order:
        rb, ra = b.get(key), a.get(key)
        out.append({"tier": key[0], "gate": key[1], "before": rb["status"] if rb else "absent",
                    "after": ra["status"] if ra else "absent", "change": change(rb, ra),
                    "before_measured": rb["measured_text"] if rb else "", "after_measured": ra["measured_text"] if ra else "",
                    "threshold": (ra or rb)["threshold"]})
    return out


def git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"(git {' '.join(args)} failed: {exc})"
    return out.stdout


def esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def text_table(table: list[dict]) -> str:
    cols = ["tier", "gate", "before", "after", "change"]
    width = {c: max(len(c), *(len(str(r[c])) for r in table)) for c in cols} if table else {c: len(c) for c in cols}
    lines = ["  ".join(c.ljust(width[c]) for c in cols), "  ".join("-" * width[c] for c in cols)]
    lines += ["  ".join(str(r[c]).ljust(width[c]) for c in cols) for r in table]
    return "\n".join(lines)


def markdown(table: list[dict], before: dict, after: dict, args: argparse.Namespace) -> str:
    repo = Path(args.repo)
    out = ["# Fix loop: before and after", ""]
    if args.before_ref and args.after_ref:
        subj = git(repo, "log", "-1", "--format=%h %s", args.after_ref).strip()
        out += [f"Before: `{args.before_ref}`. After: `{args.after_ref}` ({subj}).", ""]
    if args.rescored:
        out += [RESCORED_NOTE, ""]
    worst = (before.get("ranked_failures") or [None])[0]
    if worst:
        a = next((r for r in table if r["tier"] == worst["tier"] and r["gate"] == worst["gate"]), None)
        out += ["## Worst gate before the fix", "",
                f"{worst['tier']} {worst['gate']}: {worst['measured_text']} (threshold: {worst['threshold']}).",
                ""]
        if a:
            out += [f"After: {a['after']}, {a['after_measured']} ({a['change']}).", ""]
    counts: dict[str, int] = {}
    for r in table:
        counts[r["change"]] = counts.get(r["change"], 0) + 1
    out += ["## Gates", "", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())), ""]
    out += ["| Tier | Gate | Before | After | Change | Measured before | Measured after | Threshold |",
            "|---|---|---|---|---|---|---|---|"]
    out += [f"| {r['tier']} | {r['gate']} | {r['before']} | {r['after']} | {r['change']} | {esc(r['before_measured'])} "
            f"| {esc(r['after_measured'])} | {esc(r['threshold'])} |" for r in table]
    out.append("")
    if args.before_ref and args.after_ref:
        out += ["## Changed files", "", "```", git(repo, "diff", "--stat", args.before_ref, args.after_ref).rstrip(),
                "```", "", "## Source diff (src/)", "", "```diff",
                git(repo, "diff", args.before_ref, args.after_ref, "--", "src").rstrip(), "```", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--before-ref")
    p.add_argument("--after-ref")
    p.add_argument("--repo", default=".")
    p.add_argument("--out", type=Path)
    p.add_argument("--rescored", action="store_true", help="both runs were scored with the after ref's harness")
    args = p.parse_args(argv)
    before, after = load(args.before), load(args.after)
    table = rows(before, after)
    print(text_table(table))
    out = args.out or args.after.parent / "diff.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(table, before, after, args))
    print(f"\ndiff: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
