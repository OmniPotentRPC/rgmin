"""Command line for the censored benchmark gate."""

import argparse
import sys

from rggate.contract import ContractError, load_csv
from rggate.report import analyze, render


def _pair(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("pair must be candidate=baseline")
    candidate, baseline = text.split("=", 1)
    candidate = candidate.strip()
    baseline = baseline.strip()
    if not candidate or not baseline:
        raise argparse.ArgumentTypeError("pair must be candidate=baseline")
    return candidate, baseline


def build_parser():
    parser = argparse.ArgumentParser(prog="rggate")
    parser.add_argument("csv", help="contract v1 results")
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        type=_pair,
        help="candidate=baseline, repeatable",
    )
    parser.add_argument("--e-tol", type=float, default=1e-4)
    parser.add_argument("--rmsd-tol", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--resamples", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--plot", default=None, help="write an SVG figure")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = load_csv(args.csv)
        report = analyze(
            rows,
            args.pair,
            e_tol=args.e_tol,
            rmsd_tol=args.rmsd_tol,
            margin=args.margin,
            alpha=args.alpha,
            resamples=args.resamples,
            seed=args.seed,
            jobs=args.jobs,
        )
    except ContractError as exc:
        print(f"rggate: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"rggate: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write(render(report))
    if args.plot:
        try:
            from rggate.plot import write_svg

            write_svg(report, args.plot)
        except RuntimeError as exc:
            print(f"rggate: {exc}", file=sys.stderr)
            return 2
    return report.exit_code
