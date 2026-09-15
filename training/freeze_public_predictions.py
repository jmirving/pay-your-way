from __future__ import annotations

import argparse
import json
from pathlib import Path

from pay_your_way.public_filing_triage import recommend_public_filing_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("training/locked_candidates_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("training/locked_predictions_v1.json"))
    parser.add_argument("--model-commit", required=True, help="Exact Git commit containing the frozen rules")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    predictions = []
    for candidate in payload["candidates"]:
        ranked = recommend_public_filing_actions(candidate, limit=4)
        predictions.append(
            {
                "utility": candidate["utility"],
                "ranked_actions": [item.to_dict() for item in ranked],
            }
        )

    output = {
        "benchmark_id": payload["benchmark_id"],
        "status": "predictions-frozen-outcomes-unread",
        "model_commit": args.model_commit,
        "source_filing_snapshot": payload["source_filing_snapshot"],
        "predictions": predictions,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
