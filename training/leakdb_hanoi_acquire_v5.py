from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

import numpy as np

from leakdb_hanoi_acquire import DEV_IDS, hanoi_catalog
from leakdb_localize_acquire import parse_inp
from leakdb_sealed import _save_bundle


def load_frames_hanoi_resilient(
    ids: list[int],
    cache_dir: Path,
    *,
    attempts: int = 4,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    """Load scenarios individually so one transient download cannot discard the cohort."""
    from water_benchmark_hub import load

    benchmark = load("KIOS-LeakDB")
    xs: dict[int, np.ndarray] = {}
    ys: dict[int, np.ndarray] = {}
    features: list[str] | None = None

    for position, sid in enumerate(ids, 1):
        for attempt in range(1, attempts + 1):
            try:
                frame = benchmark.load_data(
                    [sid],
                    use_net1=False,
                    download_dir=str(cache_dir),
                    return_X_y=False,
                    verbose=False,
                )[sid]
                current = [
                    str(column)
                    for column in frame.columns
                    if str(column) not in {"labels", "timestamps"}
                ]
                if features is None:
                    features = current
                elif current != features:
                    raise RuntimeError(f"Hanoi scenario {sid} feature schema differs")
                xs[sid] = frame[current].to_numpy(dtype=np.float32)
                ys[sid] = frame["labels"].to_numpy(dtype=np.uint8)
                print(f"loaded Hanoi scenario {position}/{len(ids)}: {sid}", flush=True)
                break
            except Exception as error:
                if attempt == attempts:
                    raise RuntimeError(
                        f"Failed to load Hanoi scenario {sid} after {attempts} attempts"
                    ) from error
                delay = min(20, 3 * attempt)
                print(
                    f"retry Hanoi scenario {sid}: attempt {attempt}/{attempts} failed "
                    f"({type(error).__name__}: {error}); sleeping {delay}s",
                    flush=True,
                )
                time.sleep(delay)

    return xs, ys, features or []


def burned_hanoi_ids(leakdb_dir: Path, through: int = 4) -> list[int]:
    ids: set[int] = set()
    for number in range(1, through + 1):
        payload = json.loads(
            (leakdb_dir / f"hanoi_{number:03d}_prediction_freeze.json").read_text(encoding="utf-8")
        )
        ids.update(int(v) for v in payload["holdout_scenario_ids"])
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sealed Hanoi four-hour localization-refresh benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_v5_work"))
    parser.add_argument("--holdout-count", type=int, default=80)
    parser.add_argument("--challenge-id", default="leakdb-hanoi-pipeline-005")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    leakdb_dir = Path(__file__).resolve().parent / "leakdb"

    burned = burned_hanoi_ids(leakdb_dir, 4)
    localizer_ids = sorted(set(DEV_IDS) | set(burned))
    candidates = [sid for sid in range(101, 1001) if sid not in set(localizer_ids)]
    if len(candidates) < args.holdout_count:
        raise RuntimeError("Not enough fresh Hanoi scenarios remain")
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))

    print(
        f"Hanoi v5: loading {len(localizer_ids)} burned/development + "
        f"{len(holdout_ids)} fresh holdout scenarios"
    )
    dev_x, dev_y, features = load_frames_hanoi_resilient(localizer_ids, cache)
    hold_x, hold_y, hold_features = load_frames_hanoi_resilient(holdout_ids, cache)
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
        "task": "Hanoi end-to-end fixed incidents with four-hour localization refresh",
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
        "localization_refresh_steps": 8,
        "localization_refresh_hours": 4.0,
        "actionable_deadline_steps_after_event": 20,
        "actionable_deadline_hours_after_event": 10.0,
        "stored_ranking_depth": 5,
        "withheld_information": "fresh holdout leak timing and node IDs",
        "change_under_test": "localization refresh cadence only: six hours -> four hours, plus newly revealed events added to localizer training"
    }
    (public / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
