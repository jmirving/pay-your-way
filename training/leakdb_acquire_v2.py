from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np

from leakdb_sealed import _save_bundle

DEFAULT_DEV_IDS = list(range(1, 21)) + [155,163,209,245,332,401,430,437,512,526,555,577,706,726,730,772,837,872,923,934]


def load_frames(ids: list[int], cache_dir: Path):
    from water_benchmark_hub import load
    benchmark = load("KIOS-LeakDB")
    frames = benchmark.load_data(
        ids, use_net1=True, download_dir=str(cache_dir), return_X_y=False, verbose=False
    )
    xs: dict[int, object] = {}
    ys: dict[int, object] = {}
    features: list[str] | None = None
    for sid in ids:
        frame = frames[sid]
        current = [str(c) for c in frame.columns if str(c) not in {"labels", "timestamps"}]
        if features is None:
            features = current
        elif current != features:
            raise RuntimeError(f"Scenario {sid} feature schema differs")
        xs[sid] = frame[current].to_numpy(dtype=np.float32)
        ys[sid] = frame["labels"].to_numpy(dtype=np.uint8)
    return xs, ys, features or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("challenge_work"))
    parser.add_argument("--holdout-count", type=int, default=30)
    parser.add_argument("--holdout-min", type=int, default=101)
    parser.add_argument("--holdout-max", type=int, default=1000)
    parser.add_argument("--challenge-id", default="leakdb-net1-sealed-002")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    dev_ids = sorted(set(DEFAULT_DEV_IDS))
    candidates = [sid for sid in range(args.holdout_min, args.holdout_max + 1) if sid not in set(dev_ids)]
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    print(f"Preparing {len(dev_ids)} development and {len(holdout_ids)} sealed holdout scenarios.")
    dev_x, dev_y, features = load_frames(dev_ids, cache)
    hold_x, hold_y, hold_features = load_frames(holdout_ids, cache)
    if hold_features != features:
        raise RuntimeError("Development and holdout feature schemas differ")

    _save_bundle(out / "dev_bundle.npz", dev_ids, dev_x, dev_y, features)
    _save_bundle(out / "holdout_inputs.npz", holdout_ids, hold_x, None, features)

    labels: dict[str, object] = {"scenario_ids": np.asarray(holdout_ids, dtype=np.int32)}
    for sid in holdout_ids:
        labels[f"y_{sid}"] = np.asarray(hold_y[sid], dtype=np.uint8)
    np.savez_compressed(secret / "labels.npz", **labels)

    manifest = {
        "challenge_id": args.challenge_id,
        "benchmark": "KIOS-LeakDB",
        "network": "Net1",
        "development_scenario_ids": dev_ids,
        "development_note": "Includes fixed scenarios 1-20 plus challenge 001 after its labels were revealed.",
        "holdout_scenario_ids": holdout_ids,
        "holdout_selection": "cryptographically random sample at workflow runtime from non-development scenario IDs",
        "holdout_range": [args.holdout_min, args.holdout_max],
        "label_boundary": "holdout labels stored only in sealed-leakdb-labels artifact",
        "feature_count": len(features),
        "acquisition_workaround": "WaterBenchmarkHub 0.4.0 DataFrame path used because return_X_y=True is broken for LeakDB Net1",
    }
    (public / "challenge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
