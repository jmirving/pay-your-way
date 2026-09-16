from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from leakdb_acquire_v2 import load_frames
from leakdb_acquire_v3 import burned_ids
from leakdb_localize_acquire import leak_catalog, parse_inp
from leakdb_sealed import _save_bundle


def localization_burned_ids(leakdb_dir: Path, through: int) -> list[int]:
    ids: set[int] = set()
    for number in range(1, through + 1):
        path = leakdb_dir / f"localization_{number:03d}_prediction_freeze.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids.update(int(v) for v in payload["holdout_scenario_ids"])
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sealed end-to-end LeakDB detection+localization benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline_work"))
    parser.add_argument("--holdout-count", type=int, default=60)
    parser.add_argument("--detection-burned-through", type=int, default=5)
    parser.add_argument("--localization-burned-through", type=int, default=1)
    parser.add_argument("--challenge-id", default="leakdb-net1-pipeline-001")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    leakdb_dir = Path(__file__).resolve().parent / "leakdb"

    detection_burned, _ = burned_ids(leakdb_dir, args.detection_burned_through)
    localization_burned = localization_burned_ids(leakdb_dir, args.localization_burned_through)
    all_burned = sorted(set(detection_burned).union(localization_burned))

    # Exact v5 detector development universe ended at detection challenge 004.
    detector_dev_ids, _ = burned_ids(leakdb_dir, 4)
    localizer_dev_ids = all_burned

    candidates = [sid for sid in range(101, 1001) if sid not in set(all_burned)]
    if len(candidates) < args.holdout_count:
        raise RuntimeError("Not enough fresh scenarios remain for pipeline holdout")
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    needed_dev_ids = sorted(set(detector_dev_ids).union(localizer_dev_ids))
    print(f"Pipeline: loading {len(needed_dev_ids)} burned development + {len(holdout_ids)} fresh holdout scenarios")
    dev_x, dev_y, features = load_frames(needed_dev_ids, cache)
    hold_x, hold_y, hold_features = load_frames(holdout_ids, cache)
    if features != hold_features:
        raise RuntimeError("Development and holdout feature schemas differ")

    _save_bundle(out / "dev_bundle.npz", needed_dev_ids, dev_x, dev_y, features)
    _save_bundle(out / "holdout_inputs.npz", holdout_ids, hold_x, None, features)

    catalog = leak_catalog()
    from water_benchmark_hub import load
    network = load("Network-Net1")
    inp_path = Path(network.load(download_dir=str(cache), verbose=False, return_scenario=False))
    topology = parse_inp(inp_path)
    candidate_set = set(topology["candidate_nodes"])
    (public / "topology.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")

    # Development location labels are burned and may be used for training.
    localizer_events = []
    for sid in localizer_dev_ids:
        for event in catalog.get(str(sid), []):
            if event["node_id"] in candidate_set:
                localizer_events.append({
                    "event_id": f"{sid}:{event['event_index']}",
                    "scenario_id": sid,
                    **event,
                })
    (out / "localizer_dev_labels.json").write_text(json.dumps(localizer_events, indent=2) + "\n", encoding="utf-8")

    # Holdout truth remains secret: binary leak windows came from the source data;
    # location/event metadata come from the canonical LeakDB catalog.
    secret_scenarios = []
    for sid in holdout_ids:
        events = [
            {
                "event_id": f"{sid}:{event['event_index']}",
                "scenario_id": sid,
                **event,
            }
            for event in catalog.get(str(sid), [])
            if event["node_id"] in candidate_set
        ]
        secret_scenarios.append({"scenario_id": sid, "events": events})
    (secret / "pipeline_labels.json").write_text(json.dumps(secret_scenarios, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "challenge_id": args.challenge_id,
        "task": "end-to-end early leak alert plus node localization",
        "benchmark": "KIOS-LeakDB",
        "network": "Net1",
        "detector": "reproduce v5 detector trained/calibrated only on detection challenges 001-004",
        "localizer": "coordinate localizer trained on all burned detection scenarios through 005 plus localization challenge 001",
        "detection_development_count": len(detector_dev_ids),
        "localization_development_count": len(localizer_dev_ids),
        "localization_development_event_count": len(localizer_events),
        "holdout_scenario_ids": holdout_ids,
        "holdout_count": len(holdout_ids),
        "candidate_node_count": len(topology["candidate_nodes"]),
        "public_information": "holdout raw sensor streams and network topology only",
        "withheld_information": "holdout leak event timing and node IDs",
        "localization_observation_after_alert_steps": 10,
        "strict_early_window_steps": 10
    }
    (public / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
