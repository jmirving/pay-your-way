from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from leakdb_hanoi_acquire import DEV_IDS, hanoi_catalog, load_frames_hanoi
from leakdb_localize_acquire import parse_inp
from leakdb_sealed import _save_bundle


def previous_holdout_ids(leakdb_dir: Path) -> list[int]:
    payload = json.loads((leakdb_dir / "hanoi_001_prediction_freeze.json").read_text(encoding="utf-8"))
    return [int(v) for v in payload["holdout_scenario_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fresh sealed Hanoi incident-state benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_v2_work"))
    parser.add_argument("--holdout-count", type=int, default=40)
    parser.add_argument("--challenge-id", default="leakdb-hanoi-pipeline-002")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    leakdb_dir = Path(__file__).resolve().parent / "leakdb"

    previous = set(previous_holdout_ids(leakdb_dir))
    candidates = [sid for sid in range(101, 1001) if sid not in previous]
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    print(f"Hanoi v2: loading {len(DEV_IDS)} fixed development + {len(holdout_ids)} fresh holdout scenarios")
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
                dev_events.append({"event_id": f"{sid}:{event['event_index']}", "scenario_id": sid, **event})
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
        "task": "fresh Hanoi end-to-end benchmark with incident deduplication",
        "benchmark": "KIOS-LeakDB",
        "network": "Hanoi",
        "development_scenario_ids": DEV_IDS,
        "excluded_hanoi_001_holdout_ids": sorted(previous),
        "holdout_scenario_ids": holdout_ids,
        "holdout_count": len(holdout_ids),
        "candidate_node_count": len(topology["candidate_nodes"]),
        "raw_sensor_count": len(features),
        "detector_policy": {"threshold": 0.775, "smooth_window": 2, "source": "Hanoi pipeline 001"},
        "incident_recovery_steps": 12,
        "incident_recovery_hours": 6.0,
        "localization_observation_after_alert_steps": 10,
        "withheld_information": "holdout leak timing and node IDs",
        "change_under_test": "incident state machine only; detector trigger, training data, and localizer family remain unchanged"
    }
    (public / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
