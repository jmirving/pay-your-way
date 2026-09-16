from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np

from leakdb_sealed import _save_bundle
from leakdb_acquire_v2 import load_frames


def burned_ids(leakdb_dir: Path, burned_through: int) -> tuple[list[int], list[str]]:
    ids = set(range(1, 21))
    sources: list[str] = []
    for number in range(1, burned_through + 1):
        path = leakdb_dir / f"challenge_{number:03d}_prediction_freeze.json"
        if not path.exists():
            raise FileNotFoundError(f"Required burned-scenario freeze is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids.update(int(v) for v in payload["holdout_scenario_ids"])
        sources.append(path.name)
    return sorted(ids), sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fresh sealed LeakDB challenge using only previously burned scenarios for development.")
    parser.add_argument("--out-dir", type=Path, default=Path("challenge_work"))
    parser.add_argument("--holdout-count", type=int, default=40)
    parser.add_argument("--holdout-min", type=int, default=101)
    parser.add_argument("--holdout-max", type=int, default=1000)
    parser.add_argument("--burned-through", type=int, default=2)
    parser.add_argument("--challenge-id", default="leakdb-net1-sealed-003")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    leakdb_dir = Path(__file__).resolve().parent / "leakdb"
    dev_ids, freeze_sources = burned_ids(leakdb_dir, args.burned_through)
    dev_set = set(dev_ids)
    candidates = [sid for sid in range(args.holdout_min, args.holdout_max + 1) if sid not in dev_set]
    if len(candidates) < args.holdout_count:
        raise ValueError("Not enough unburned scenarios remain for requested holdout")
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    overlap = dev_set.intersection(holdout_ids)
    if overlap:
        raise RuntimeError(f"Sealed holdout overlaps burned development data: {sorted(overlap)}")

    print(f"Preparing {len(dev_ids)} burned development and {len(holdout_ids)} fresh sealed holdout scenarios.")
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
        "development_count": len(dev_ids),
        "burned_through_challenge": args.burned_through,
        "burned_freeze_sources": freeze_sources,
        "holdout_scenario_ids": holdout_ids,
        "holdout_count": len(holdout_ids),
        "holdout_selection": "cryptographically random sample at workflow runtime from scenario IDs not present in any burned freeze through the configured challenge",
        "holdout_range": [args.holdout_min, args.holdout_max],
        "label_boundary": "holdout labels stored only in sealed-leakdb-labels artifact; prediction process receives holdout_inputs.npz only",
        "feature_count": len(features),
        "acquisition_workaround": "WaterBenchmarkHub 0.4.0 DataFrame path used because return_X_y=True is broken for LeakDB Net1",
    }
    (public / "challenge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
