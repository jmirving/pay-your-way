from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakdb_hanoi_acquire import TRAIN_IDS
from leakdb_hanoi_predict_v3 import hysteresis_incidents
from leakdb_localize_predict import event_signature, rank_nodes
from leakdb_predict_v2 import causal_smooth
from leakdb_predict_v3 import fit_model, raw_probabilities
from leakdb_sealed import _load_bundle

OPEN_SMOOTH = 2
CLOSE_THRESHOLD = 0.5
RECOVERY_STEPS = 96
LOCALIZATION_WINDOW_STEPS = 10
LOCALIZATION_REFRESH_STEPS = 12


def snapshot_signature(x: np.ndarray, ready_idx: int) -> np.ndarray:
    start = max(0, ready_idx - LOCALIZATION_WINDOW_STEPS + 1)
    horizon = ready_idx - start + 1
    return event_signature(x, start, horizon=horizon)


def localization_ready_indices(start: int, end: int, n_steps: int) -> list[int]:
    first = min(n_steps - 1, start + LOCALIZATION_WINDOW_STEPS - 1)
    ready = [first]
    nxt = first + LOCALIZATION_REFRESH_STEPS
    while nxt <= end and nxt < n_steps:
        ready.append(nxt)
        nxt += LOCALIZATION_REFRESH_STEPS
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed Hanoi incident policy with dynamic localization refresh.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_v4_work"))
    args = parser.parse_args()

    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import ExtraTreesRegressor

    out = args.out_dir
    public = out / "public"
    dev_ids, dev_x, dev_y, features = _load_bundle(out / "dev_bundle.npz")
    hold_ids, hold_x, _, hold_features = _load_bundle(out / "holdout_inputs.npz")
    if dev_y is None or features != hold_features:
        raise RuntimeError("Hanoi development labels missing or feature schema mismatch")

    detector = fit_model(dev_x, dev_y, TRAIN_IDS)

    topology = json.loads((public / "topology.json").read_text(encoding="utf-8"))
    coords = topology["coordinates"]
    candidates = topology["candidate_nodes"]
    localizer_events = json.loads(
        (out / "localizer_dev_labels.json").read_text(encoding="utf-8")
    )
    localizer_events = [e for e in localizer_events if e["node_id"] in coords]
    loc_x = np.stack([
        event_signature(dev_x[e["scenario_id"]], e["start_idx"])
        for e in localizer_events
    ])
    loc_y = np.asarray([coords[e["node_id"]] for e in localizer_events], dtype=np.float64)
    localizer = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=0,
        n_jobs=-1,
    )
    localizer.fit(loc_x, loc_y)

    results = []
    total_snapshots = 0
    for sid in hold_ids:
        x = hold_x[sid]
        probability = causal_smooth(raw_probabilities(detector, x), OPEN_SMOOTH)
        raw_episodes = hysteresis_incidents(probability, CLOSE_THRESHOLD, RECOVERY_STEPS)

        episode_specs = []
        signatures = []
        for episode_idx, (start, end) in enumerate(raw_episodes):
            for ready_idx in localization_ready_indices(start, end, len(x)):
                episode_specs.append((episode_idx, start, end, ready_idx))
                signatures.append(snapshot_signature(x, ready_idx))

        predicted_xy = (
            localizer.predict(np.stack(signatures))
            if signatures
            else np.empty((0, 2), dtype=np.float64)
        )

        snapshots_by_episode: list[list[dict]] = [[] for _ in raw_episodes]
        for spec, xy in zip(episode_specs, predicted_xy):
            episode_idx, start, end, ready_idx = spec
            ranking = rank_nodes(xy, coords, candidates)
            snapshots_by_episode[episode_idx].append({
                "localization_ready_idx": ready_idx,
                "window_start_idx": max(0, ready_idx - LOCALIZATION_WINDOW_STEPS + 1),
                "predicted_xy": [float(xy[0]), float(xy[1])],
                "ranked_nodes": ranking,
            })

        episodes = []
        for episode_idx, (start, end) in enumerate(raw_episodes):
            snapshots = snapshots_by_episode[episode_idx]
            if not snapshots:
                continue
            initial = snapshots[0]
            episodes.append({
                "alert_start_idx": start,
                "alert_end_idx": end,
                "localization_ready_idx": initial["localization_ready_idx"],
                "predicted_xy": initial["predicted_xy"],
                "ranked_nodes": initial["ranked_nodes"],
                "localization_snapshots": snapshots,
            })
            total_snapshots += len(snapshots)

        results.append({
            "scenario_id": sid,
            "timesteps": int(len(x)),
            "alert_episodes": episodes,
        })

    (public / "pipeline_predictions.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "algorithm": "Pay Your Way LeakDB Hanoi pipeline v4 dynamic-localization",
        "implementation": "batched localization inference; mathematically equivalent to v4 per-snapshot inference",
        "detector": {
            "family": "fixed Hanoi HGB detector",
            "training_scenario_ids": TRAIN_IDS,
            "open_threshold": 0.775,
            "open_smooth_window": OPEN_SMOOTH,
        },
        "incident_state": {
            "close_threshold": CLOSE_THRESHOLD,
            "recovery_steps": RECOVERY_STEPS,
            "recovery_hours": RECOVERY_STEPS * 0.5,
            "source": "frozen Hanoi pipeline-003 policy",
        },
        "localizer": {
            "family": "ExtraTrees coordinate localizer with expanded burned-event training",
            "training_scenario_count": len(dev_ids),
            "training_event_count": len(localizer_events),
            "candidate_node_count": len(candidates),
            "window_steps": LOCALIZATION_WINDOW_STEPS,
            "refresh_steps": LOCALIZATION_REFRESH_STEPS,
            "refresh_hours": LOCALIZATION_REFRESH_STEPS * 0.5,
            "behavior": "rankings refresh causally from the latest five hours while an incident remains open",
            "total_holdout_snapshots": total_snapshots,
        },
        "holdout_scenario_count": len(hold_ids),
        "raw_sensor_count": len(features or []),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No Hanoi-004 holdout leak timing or node labels are opened by prediction.",
    }
    (public / "pipeline_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Hanoi v4 predicted {len(hold_ids)} sealed scenarios with "
        f"{sum(len(r['alert_episodes']) for r in results)} incidents and "
        f"{total_snapshots} batched localization snapshots"
    )


if __name__ == "__main__":
    main()
