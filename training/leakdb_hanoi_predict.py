from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakdb_hanoi_acquire import TRAIN_IDS, VALIDATION_A, VALIDATION_B
from leakdb_localize_predict import event_signature, rank_nodes
from leakdb_pipeline_predict_v2 import choose_policy, make_alert
from leakdb_predict_v3 import fit_model, raw_probabilities
from leakdb_sealed import _load_bundle

LOCALIZATION_OBSERVATION_STEPS = 10


def alert_episode_starts(alert: np.ndarray) -> list[int]:
    alert = np.asarray(alert, dtype=np.uint8)
    return [int(v) for v in np.flatnonzero((alert == 1) & (np.r_[0, alert[:-1]] == 0))]


def alert_episode_end(alert: np.ndarray, start: int) -> int:
    i = start
    while i + 1 < len(alert) and alert[i + 1] == 1:
        i += 1
    return i


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/calibrate the Pay Your Way method on LeakDB Hanoi and predict a sealed holdout.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_work"))
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
    validation_ids = sorted(set(VALIDATION_A + VALIDATION_B))
    raw_val = {sid: raw_probabilities(detector, dev_x[sid]) for sid in validation_ids}
    groups = {"hanoi_val_a": VALIDATION_A, "hanoi_val_b": VALIDATION_B}
    (smooth, threshold, hold), calibration = choose_policy(dev_y, groups, raw_val)

    topology = json.loads((public / "topology.json").read_text(encoding="utf-8"))
    coords = topology["coordinates"]
    candidates = topology["candidate_nodes"]
    localizer_events = json.loads((out / "localizer_dev_labels.json").read_text(encoding="utf-8"))
    localizer_events = [e for e in localizer_events if e["node_id"] in coords]
    if not localizer_events:
        raise RuntimeError("No coordinate-resolvable Hanoi development leak events")
    loc_x = np.stack([event_signature(dev_x[e["scenario_id"]], e["start_idx"]) for e in localizer_events])
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
    for sid in hold_ids:
        x = hold_x[sid]
        raw = raw_probabilities(detector, x)
        alert = make_alert(raw, smooth, threshold, hold)
        episodes = []
        for start in alert_episode_starts(alert):
            end = alert_episode_end(alert, start)
            sig = event_signature(x, start, horizon=LOCALIZATION_OBSERVATION_STEPS)
            pred_xy = localizer.predict(sig.reshape(1, -1))[0]
            ranking = rank_nodes(pred_xy, coords, candidates)
            episodes.append({
                "alert_start_idx": start,
                "alert_end_idx": end,
                "localization_ready_idx": min(len(x) - 1, start + LOCALIZATION_OBSERVATION_STEPS - 1),
                "predicted_xy": [float(pred_xy[0]), float(pred_xy[1])],
                "ranked_nodes": ranking,
            })
        results.append({
            "scenario_id": sid,
            "timesteps": int(len(x)),
            "alert_positive_steps": int(np.sum(alert)),
            "alert_episodes": episodes,
        })

    (public / "pipeline_predictions.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = {
        "algorithm": "Pay Your Way LeakDB Hanoi pipeline v1",
        "detector": {
            "family": "same HGB + robust new-alert calibration family used on Net1",
            "training_scenario_ids": TRAIN_IDS,
            "validation_groups": groups,
            "alert_threshold": threshold,
            "alert_smooth_window": smooth,
            "alert_hold_steps": hold,
            "calibration_results": calibration,
        },
        "localizer": {
            "family": "ExtraTrees coordinate-regression localizer, same feature design as Net1",
            "training_event_count": len(localizer_events),
            "observation_steps_after_alert": LOCALIZATION_OBSERVATION_STEPS,
            "candidate_node_count": len(candidates),
        },
        "holdout_scenario_count": len(hold_ids),
        "raw_sensor_count": len(features or []),
        "signature_feature_count": int(loc_x.shape[1]),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No Hanoi holdout leak timing or node labels are opened by prediction.",
        "transfer_boundary": "No Net1 trained model or Net1 coordinate model is used; only the algorithm family/feature definitions transfer."
    }
    (public / "pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Hanoi predicted {len(hold_ids)} sealed scenarios; candidates={len(candidates)}; "
        f"policy threshold={threshold:.3f}, smooth={smooth}, hold={hold}"
    )


if __name__ == "__main__":
    main()
