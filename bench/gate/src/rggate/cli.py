"""Command line for the censored benchmark gate."""

import argparse
import sys

from rggate.contract import ContractError, load_csv
from rggate.report import analyze, render


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        self.exit(3)


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
    parser = _Parser(prog="rggate")
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
    parser.add_argument("--reference", choices=("auto", "lowest"), default="auto")
    parser.add_argument("--plan-n", type=int, default=50)
    parser.add_argument("--min-clusters", type=int, default=10)
    parser.add_argument(
        "--boundary",
        choices=("normal-mixture", "agrapa"),
        default="normal-mixture",
        help="normal-mixture, or agrapa (one signed stake, c=3/4)",
    )
    parser.add_argument("--plot", default=None, help="write an SVG figure")
    return parser


def main(argv=None):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        return int(exc.code)
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
            reference=args.reference,
            plan_n=args.plan_n,
            boundary=args.boundary,
            min_clusters=args.min_clusters,
        )
    except ContractError as exc:
        print(f"rggate: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"rggate: {exc}", file=sys.stderr)
        return 3
    sys.stdout.write(render(report))
    if args.plot:
        try:
            from rggate.plot import write_svg

            write_svg(report, args.plot)
        except RuntimeError as exc:
            print(f"rggate: {exc}", file=sys.stderr)
            return 3
    return report.exit_code
