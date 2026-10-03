"""Command line: scan2scope run | bench | fetch-weights | doctor | synth."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from scan2scope.pipeline import run_capture

    path = Path(args.capture).expanduser()
    out = Path(args.out).expanduser() if args.out else Path("out") / path.stem.replace(" ", "_")
    run_capture(path, out, tier=args.tier, cache_mode=args.cache, drift_correction=not args.no_drift,
                semantics=not args.no_semantics)
    print(f"\nresult: {out / 'result.json'}\nplan:   {out / 'plan.svg'}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from scan2scope.bench.runner import run_benchmark

    run_benchmark(Path(args.data).expanduser(), Path(args.out).expanduser(), cache_mode=args.cache,
                  only=args.only, skip_run=args.skip_run, semantics=not args.no_semantics)
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from scan2scope.weights import fetch_all

    fetch_all(verify=not args.no_verify)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from scan2scope.weights import doctor

    return doctor()


def _cmd_synth(args: argparse.Namespace) -> int:
    from scan2scope.synth.generate import generate_benchmark

    generate_benchmark(Path(args.out).expanduser(), n_properties=args.properties, seed=args.seed)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scan2scope", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process one capture (photo folders, a video file, or a Stray Scanner export)")
    r.add_argument("capture")
    r.add_argument("--out", help="output directory (default: out/<capture name>)")
    r.add_argument("--tier", choices=["photo", "video", "lidar"], help="override tier detection")
    r.add_argument("--cache", choices=["live", "replay", "off"], default="live",
                   help="live: compute and store model outputs; replay: reuse stored outputs only")
    r.add_argument("--no-drift", action="store_true", help="disable drift correction (ablation)")
    r.add_argument("--no-semantics", action="store_true", help="skip damage and object detection")
    r.set_defaults(fn=_cmd_run)

    b = sub.add_parser("bench", help="run and score the benchmark")
    b.add_argument("data", help="benchmark data root (one folder per property with ground_truth.yaml)")
    b.add_argument("--out", default="runs/bench")
    b.add_argument("--cache", choices=["live", "replay", "off"], default="live")
    b.add_argument("--only", nargs="*", help="capture ids to run")
    b.add_argument("--skip-run", action="store_true", help="score existing results without rerunning")
    b.add_argument("--no-semantics", action="store_true", help="skip damage detection (synthetic captures)")
    b.set_defaults(fn=_cmd_bench)

    f = sub.add_parser("fetch-weights", help="download pinned model weights")
    f.add_argument("--no-verify", action="store_true")
    f.set_defaults(fn=_cmd_fetch)

    d = sub.add_parser("doctor", help="check weights, device and disk before a run")
    d.set_defaults(fn=_cmd_doctor)

    s = sub.add_parser("synth", help="generate synthetic LiDAR benchmark captures with exact ground truth")
    s.add_argument("--out", default="bench/synthetic")
    s.add_argument("--properties", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_synth)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
