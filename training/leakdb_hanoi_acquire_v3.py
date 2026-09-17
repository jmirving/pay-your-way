from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from leakdb_hanoi_acquire import DEV_IDS, hanoi_catalog, load_frames_hanoi
from leakdb_localize_acquire import parse_inp
from leakdb_sealed import _save_bundle


def burned_hanoi_holdouts(leakdb_dir: Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for number in (1, 2):
        payload = json.loads(
            (leakdb_dir / f"hanoi_{number:03d}_prediction_freeze.json").read_text(encoding="utf-8")
        )
        groups[f"hanoi_{number:03d}"] = [int(v) for v in payload["holdout_scenario_ids"]]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sealed Hanoi incident-hysteresis benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_v3_work"))
    parser.add_argument("--holdout-count", type=int, default=60)
    parser.add_argument("--challenge-id", default="leakdb-hanoi-pipeline-003")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    leakdb_dir = Path(__file__).resolve().parent / "leakdb"

    groups = burned_hanoi_holdouts(leakdb_dir)
    calibration_ids = sorted({sid for values in groups.values() for sid in values})
    excluded = set(DEV_IDS) | set(calibration_ids)
    candidates = [sid for sid in range(101, 1001) if sid not in excluded]
    if len(candidates) < args.holdout_count:
        raise RuntimeError("Not enough fresh Hanoi scenarios")
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    needed_ids = sorted(set(DEV_IDS) | set(calibration_ids))
    print(
        f"Hanoi v3: loading {len(needed_ids)} dev/calibration + "
        f"{len(holdout_ids)} fresh holdout scenarios"
    )
    dev_x, dev_y, features = load_frames_hanoi(needed_ids, cache)
    hold_x, hold_y, hold_features = load_frames_hanoi(holdout_ids, cache)
    if features != hold_features:
        raise RuntimeError("Hanoi development and holdout feature schemas differ")
    _save_bundle(out / "dev_bundle.npz", needed_ids, dev_x, dev_y, features)
    _save_bundle(out / "holdout_inputs.npz", holdout_ids, hold_x, None, features)

    from water_benchmark_hub import load
    network = load("Network-Hanoi")
    inp_path = Path(network.load(download_dir=str(cache), verbose=False, return_scenario=False))
    topology = parse_inp(inp_path)
    candidate_set = set(topology["candidate_nodes"])
    (public / "topology.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")

    catalog = hanoi_catalog()
    localizer_events = []
    for sid in DEV_IDS:
        for event in catalog.get(str(sid), []):
            if event["node_id"] in candidate_set:
                localizer_events.append({
                    "event_id": f"{sid}:{event['event_index']}",
                    "scenario_id": sid,
                    **event,
                })
    (out / "localizer_dev_labels.json").write_text(
        json.dumps(localizer_events, indent=2) + "\n", encoding="utf-8"
    )

    calibration_labels = {}
    for name, ids in groups.items():
        rows = []
        for sid in ids:
            rows.append({
                "scenario_id": sid,
                "events": [
                    {"event_id": f"{sid}:{event['event_index']}", "scenario_id": sid, **event}
                    for event in catalog.get(str(sid), [])
                    if event["node_id"] in candidate_set
                ],
            })
        calibration_labels[name] = rows
    (out / "incident_calibration_labels.json").write_text(
        json.dumps(calibration_labels, indent=2) + "\n", encoding="utf-8"
    )

    secret_rows = []
    for sid in holdout_ids:
        events = [
            {"event_id": f"{sid}:{event['event_index']}", "scenario_id": sid, **event}
            for event in catalog.get(str(sid), [])
            if event["node_id"] in candidate_set
        ]
        secret_rows.append({"scenario_id": sid, "events": events})
    (secret / "pipeline_labels.json").write_text(
        json.dumps(secret_rows, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "challenge_id": args.challenge_id,
        "task": "fresh Hanoi end-to-end benchmark with calibrated incident closure hysteresis",
        "benchmark": "KIOS-LeakDB",
        "network": "Hanoi",
        "fixed_detector_training_scenario_ids": list(range(1, 41)),
        "fixed_localizer_development_scenario_ids": DEV_IDS,
        "incident_calibration_groups": groups,
        "holdout_scenario_ids": holdout_ids,
        "holdout_count": len(holdout_ids),
        "candidate_node_count": len(topology["candidate_nodes"]),
        "raw_sensor_count": len(features),
        "open_policy": {"threshold": 0.775, "smooth_window": 2, "source": "Hanoi pipeline 001"},
        "close_policy_search": {
            "close_thresholds": [0.2, 0.3, 0.4, 0.5, 0.6],
            "recovery_steps": [12, 24, 48, 96, 192],
            "recovery_hours": [6, 12, 24, 48, 96]
        },
        "localization_observation_after_alert_steps": 10,
        "withheld_information": "fresh holdout leak timing and node IDs",
        "change_under_test": "incident close hysteresis only; detector opening rule and localizer are fixed"
    }
    (public / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
