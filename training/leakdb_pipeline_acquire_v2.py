from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from leakdb_acquire_v2 import load_frames
from leakdb_acquire_v3 import burned_ids
from leakdb_localize_acquire import leak_catalog, parse_inp
from leakdb_pipeline_acquire import localization_burned_ids
from leakdb_sealed import _save_bundle


def freeze_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(v) for v in payload["holdout_scenario_ids"]]


def challenge_ids(leakdb_dir: Path, number: int) -> list[int]:
    return freeze_ids(leakdb_dir / f"challenge_{number:03d}_prediction_freeze.json")


def pipeline_burned_ids(leakdb_dir: Path, through: int) -> list[int]:
    ids: set[int] = set()
    for number in range(1, through + 1):
        ids.update(freeze_ids(leakdb_dir / f"pipeline_{number:03d}_prediction_freeze.json"))
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare sealed end-to-end LeakDB pipeline v2 benchmark."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline_v2_work"))
    parser.add_argument("--holdout-count", type=int, default=80)
    parser.add_argument("--challenge-id", default="leakdb-net1-pipeline-002")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    leakdb_dir = Path(__file__).resolve().parent / "leakdb"

    detection_burned, _ = burned_ids(leakdb_dir, 5)
    localization_burned = localization_burned_ids(leakdb_dir, 1)
    pipeline_001_ids = pipeline_burned_ids(leakdb_dir, 1)
    all_burned = sorted(
        set(detection_burned).union(localization_burned).union(pipeline_001_ids)
    )

    # Keep the localizer frozen to the pipeline-001 training universe.
    localizer_dev_ids = sorted(set(detection_burned).union(localization_burned))

    # Detector policy gets disjoint train/validation slices from each burned
    # detection challenge; pipeline 001 is validation-only.
    base_train_ids = list(range(1, 21))
    detector_train_ids: set[int] = set(base_train_ids)
    detector_validation_groups: dict[str, list[int]] = {}
    for number in range(1, 6):
        ids = sorted(challenge_ids(leakdb_dir, number))
        val = ids[::2]
        train = ids[1::2]
        detector_validation_groups[f"challenge_{number:03d}"] = val
        detector_train_ids.update(train)
    detector_validation_groups["pipeline_001"] = sorted(pipeline_001_ids)

    needed_dev_ids = sorted(
        detector_train_ids
        .union(*(set(v) for v in detector_validation_groups.values()))
        .union(localizer_dev_ids)
    )

    burned_set = set(all_burned).union(base_train_ids)
    candidates = [sid for sid in range(101, 1001) if sid not in burned_set]
    if len(candidates) < args.holdout_count:
        raise RuntimeError("Not enough fresh scenarios remain for pipeline-v2 holdout")
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    print(
        f"Pipeline v2: loading {len(needed_dev_ids)} development + "
        f"{len(holdout_ids)} fresh holdout scenarios"
    )
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
    (public / "topology.json").write_text(
        json.dumps(topology, indent=2) + "\n", encoding="utf-8"
    )

    localizer_events = []
    for sid in localizer_dev_ids:
        for event in catalog.get(str(sid), []):
            if event["node_id"] in candidate_set:
                localizer_events.append(
                    {
                        "event_id": f"{sid}:{event['event_index']}",
                        "scenario_id": sid,
                        **event,
                    }
                )
    (out / "localizer_dev_labels.json").write_text(
        json.dumps(localizer_events, indent=2) + "\n", encoding="utf-8"
    )

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
    (secret / "pipeline_labels.json").write_text(
        json.dumps(secret_scenarios, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "challenge_id": args.challenge_id,
        "task": "end-to-end early leak alert plus node localization",
        "benchmark": "KIOS-LeakDB",
        "network": "Net1",
        "detector_method_change": (
            "Train on disjoint slices from detection challenges 001-005; calibrate "
            "new-alert episode starts on held-out halves plus pipeline 001, then test once "
            "on a completely fresh cohort."
        ),
        "detector_training_scenario_ids": sorted(detector_train_ids),
        "detector_validation_groups": detector_validation_groups,
        "localizer": (
            "Frozen pipeline-001 ExtraTrees coordinate localizer trained on detection "
            "challenges 001-005 plus localization challenge 001."
        ),
        "localization_development_count": len(localizer_dev_ids),
        "localization_development_event_count": len(localizer_events),
        "pipeline_001_burned_count": len(pipeline_001_ids),
        "holdout_scenario_ids": holdout_ids,
        "holdout_count": len(holdout_ids),
        "candidate_node_count": len(topology["candidate_nodes"]),
        "public_information": "holdout raw sensor streams and network topology only",
        "withheld_information": "holdout leak event timing and node IDs",
        "localization_observation_after_alert_steps": 10,
        "strict_early_window_steps": 10,
    }
    (public / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
