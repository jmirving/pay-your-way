from __future__ import annotations

import argparse
import sys

from .water_audit import AuditInputError, format_report, load_csv, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pay-your-way",
        description="Screen small water-system data for measurable water, energy, cost, and carbon opportunities.",
    )
    parser.add_argument("csv", help="Monthly water-system CSV using the canonical schema")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text")
    args = parser.parse_args(argv)

    try:
        records = load_csv(args.csv)
        summary, _ = summarize(records)
    except (OSError, AuditInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(summary.to_json() if args.json else format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
