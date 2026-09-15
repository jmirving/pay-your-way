from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("twdb_2021_development_findings.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    counter = Counter(
        category
        for case in payload["cases"]
        for category in case["finding_categories"]
    )
    total = len(payload["cases"])
    print(f"development utilities: {total}")
    for category, count in counter.most_common():
        print(f"{category}: {count}/{total} ({count / total:.0%})")


if __name__ == "__main__":
    main()
