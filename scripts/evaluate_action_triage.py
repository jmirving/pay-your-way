from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pay_your_way.action_triage import recommend_actions


def evaluate(path: Path) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    top1_hits = 0
    top3_hits = 0
    by_group: dict[str, list[bool]] = defaultdict(list)
    rows = []

    for case in cases:
        expected = set(case.get("expected_actions", []))
        if not expected:
            continue

        # Only evidence is passed to the recommender. Outcomes are deliberately hidden.
        ranked = recommend_actions(case["evidence"], limit=3)
        actions = [item.action for item in ranked]
        top1 = bool(actions and actions[0] in expected)
        top3 = any(action in expected for action in actions)

        total += 1
        top1_hits += int(top1)
        top3_hits += int(top3)
        by_group[case["case_group"]].append(top1)
        rows.append(
            {
                "case_id": case["case_id"],
                "expected": sorted(expected),
                "ranked": actions,
                "top1_hit": top1,
                "top3_hit": top3,
            }
        )

    group_top1 = {group: sum(hits) / len(hits) for group, hits in sorted(by_group.items())}

    return {
        "cases": total,
        "top1_accuracy": top1_hits / total if total else 0.0,
        "top3_recall": top3_hits / total if total else 0.0,
        "group_top1_accuracy": group_top1,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the transparent action-triage baseline on the seed case corpus."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="training/action_cases.json",
        type=Path,
        help="Path to the action case corpus.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    result = evaluate(args.path)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"cases: {result['cases']}")
    print(f"top-1 (in-sample regression): {result['top1_accuracy']:.1%}")
    print(f"top-3 (in-sample regression): {result['top3_recall']:.1%}")
    print()
    for row in result["rows"]:
        marker = "PASS" if row["top1_hit"] else "MISS"
        print(
            f"{marker:4} {row['case_id']}: expected={','.join(row['expected'])} "
            f"ranked={','.join(row['ranked'])}"
        )


if __name__ == "__main__":
    main()
