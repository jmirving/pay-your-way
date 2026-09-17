from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from leakdb_hanoi_acquire import DEV_IDS, hanoi_catalog, load_frames_hanoi
from leakdb_localize_acquire import parse_inp
from leakdb_sealed import _save_bundle


def burned_hanoi_ids(leakdb_dir: Path, through: int = 3) -> list[int]:
    ids: set[int] = set()
    for number in range(1, through + 1):
        payload = json.loads(
            (leakdb_dir / f"hanoi_{number:03d}_prediction_freeze.json").read_text(encoding="utf-8")
        )
        ids.update(int(v) for v in payload["holdout_scenario_ids"])
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sealed Hanoi dynamic-localization benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_v4_work"))
    parser.add_argument("--holdout-count", type=int, default=70)
    parser.add_argument("--challenge-id", default="leakdb-hanoi-pipeline-004")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    leakdb_dir = Path(__file__).resolve().parent / "leakdb"

    burned = burned_hanoi_ids(leakdb_dir, 3)
    localizer_ids = sorted(set(DEV_IDS) | set(burned))
    excluded = set(localizer_ids)
    candidates = [sid for sid in range(101, 1001) if sid not in excluded]
    if len(candidates) < args.holdout_count:
        raise RuntimeError("Not enough fresh Hanoi scenarios remain")
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    print(
        f"Hanoi v4: loading {len(localizer_ids)} burned/development + "
        f"{len(holdout_ids)} fresh holdout scenarios"
    )
    dev_x, dev_y, features = load_frames_hanoi(localizer_ids, cache)
    hold_x, hold_y, hold_features = load_frames_hanoi(holdout_ids, cache)
    if features != hold_features:
        raise RuntimeError("Hanoi development and holdout feature schemas differ")
    _save_bundle(out / "dev_bundle.npz", localizer_ids, dev_x, dev_y, features)
    _save_bundle(out / "holdout_inputs.npz", holdout_ids, hold_x, None, features)

    from water_benchmark_hub import load
    network = load("Network-Hanoi")
    inp_path = Path(network.load(download_dir=str(cache), verbose=False, return_scenario=False))
    topology = parse_inp(inp_path)
    candidate_set = set(topology["candidate_nodes"])
    (public / "topology.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")

    catalog = hanoi_catalog()
    localizer_events = []
    for sid in localizer_ids:
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
        "task": "Hanoi end-to-end incident detection with continuously refreshed localization",
        "benchmark": "KIOS-LeakDB",
        "network": "Hanoi",
        "detector_training_scenario_ids": list(range(1, 41)),
        "incident_policy": {
            "open_threshold": 0.775,
            "open_smooth_window": 2,
            "close_threshold": 0.5,
            "recovery_steps": 96,
            "source": "Hanoi pipeline 003"
        },
        "localizer_development_scenario_ids": localizer_ids,
        "localizer_development_event_count": len(localizer_events),
        "holdout_scenario_ids": holdout_ids,
        "holdout_count": len(holdout_ids),
        "candidate_node_count": len(topology["candidate_nodes"]),
        "raw_sensor_count": len(features),
        "localization_window_steps": 10,
        "localization_refresh_steps": 12,
        "localization_refresh_hours": 6.0,
        "actionable_deadline_steps_after_event": 20,
        "actionable_deadline_hours_after_event": 10.0,
        "withheld_information": "fresh holdout leak timing and node IDs",
        "change_under_test": "expanded burned-event localizer training plus causal localization refresh inside fixed incidents"
    }
    (public / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
