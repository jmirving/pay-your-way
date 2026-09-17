from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np

from leakdb_localize_acquire import parse_inp
from leakdb_sealed import _save_bundle

TRAIN_IDS = list(range(1, 41))
VALIDATION_A = list(range(41, 51))
VALIDATION_B = list(range(51, 61))
DEV_IDS = list(range(1, 61))


def load_frames_hanoi(ids: list[int], cache_dir: Path):
    from water_benchmark_hub import load

    benchmark = load("KIOS-LeakDB")
    frames = benchmark.load_data(
        ids, use_net1=False, download_dir=str(cache_dir), return_X_y=False, verbose=False
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
            raise RuntimeError(f"Hanoi scenario {sid} feature schema differs")
        xs[sid] = frame[current].to_numpy(dtype=np.float32)
        ys[sid] = frame["labels"].to_numpy(dtype=np.uint8)
    return xs, ys, features or []


def hanoi_catalog() -> dict[str, list[dict]]:
    from water_benchmark_hub.leakdb.leakdb_data import HANOI_LEAKAGES

    raw = json.loads(HANOI_LEAKAGES)
    result: dict[str, list[dict]] = {}
    for sid, events in raw.items():
        cleaned = []
        for index, event in enumerate(events):
            cleaned.append({
                "event_index": index,
                "node_id": str(event["node_id"]),
                "start_idx": int(event["leak_start_time"]),
                "end_idx": int(event["leak_end_time"]),
                "leak_diameter": float(event.get("leak_diameter", 0.0)),
            })
        result[str(sid)] = cleaned
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sealed LeakDB Hanoi end-to-end benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_work"))
    parser.add_argument("--holdout-count", type=int, default=30)
    parser.add_argument("--challenge-id", default="leakdb-hanoi-pipeline-001")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    candidates = [sid for sid in range(101, 1001) if sid not in set(DEV_IDS)]
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    print(f"Hanoi: loading {len(DEV_IDS)} development + {len(holdout_ids)} sealed holdout scenarios")
    dev_x, dev_y, features = load_frames_hanoi(DEV_IDS, cache)
    hold_x, hold_y, hold_features = load_frames_hanoi(holdout_ids, cache)
    if features != hold_features:
        raise RuntimeError("Hanoi development and holdout feature schemas differ")
    _save_bundle(out / "dev_bundle.npz", DEV_IDS, dev_x, dev_y, features)
    _save_bundle(out / "holdout_inputs.npz", holdout_ids, hold_x, None, features)

    from water_benchmark_hub import load
    network = load("Network-Hanoi")
    inp_path = Path(network.load(download_dir=str(cache), verbose=False, return_scenario=False))
    topology = parse_inp(inp_path)
    candidate_set = set(topology["candidate_nodes"])
    (public / "topology.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")

    catalog = hanoi_catalog()
    dev_events = []
    for sid in DEV_IDS:
        for event in catalog.get(str(sid), []):
            if event["node_id"] in candidate_set:
                dev_events.append({
                    "event_id": f"{sid}:{event['event_index']}",
                    "scenario_id": sid,
                    **event,
                })
    (out / "localizer_dev_labels.json").write_text(json.dumps(dev_events, indent=2) + "\n", encoding="utf-8")

    secret_rows = []
    for sid in holdout_ids:
        events = [
            {"event_id": f"{sid}:{event['event_index']}", "scenario_id": sid, **event}
            for event in catalog.get(str(sid), [])
            if event["node_id"] in candidate_set
        ]
        secret_rows.append({"scenario_id": sid, "events": events})
    (secret / "pipeline_labels.json").write_text(json.dumps(secret_rows, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "challenge_id": args.challenge_id,
        "task": "sealed early leak alert plus node localization on a larger topology",
        "benchmark": "KIOS-LeakDB",
        "network": "Hanoi",
        "detector_training_scenario_ids": TRAIN_IDS,
        "detector_validation_groups": {"hanoi_val_a": VALIDATION_A, "hanoi_val_b": VALIDATION_B},
        "localizer_development_scenario_ids": DEV_IDS,
        "localizer_development_event_count": len(dev_events),
        "holdout_scenario_ids": holdout_ids,
        "holdout_count": len(holdout_ids),
        "candidate_node_count": len(topology["candidate_nodes"]),
        "raw_sensor_count": len(features),
        "public_information": "holdout Hanoi sensor streams and Hanoi network topology only",
        "withheld_information": "holdout leak timing and leak node IDs",
        "strict_early_window_steps": 10,
        "localization_observation_after_alert_steps": 10,
        "transfer_boundary": "Algorithm family transfers from Net1; no Net1 model weights or Hanoi holdout outcomes are used."
    }
    (public / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
