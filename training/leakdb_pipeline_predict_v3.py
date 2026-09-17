from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakdb_localize_predict import event_signature, rank_nodes
from leakdb_predict_v2 import causal_smooth
from leakdb_predict_v3 import fit_model, latch_alert, raw_probabilities
from leakdb_sealed import _load_bundle

ALERT_THRESHOLD = 0.7
ALERT_SMOOTH = 3
ALERT_HOLD = 8
LOCALIZATION_OBSERVATION_STEPS = 10


def freeze_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(v) for v in payload["holdout_scenario_ids"]]


def challenge_ids(leakdb_dir: Path, number: int) -> list[int]:
    return sorted(freeze_ids(leakdb_dir / f"challenge_{number:03d}_prediction_freeze.json"))


def alert_episode_starts(alert: np.ndarray) -> list[int]:
    alert = np.asarray(alert, dtype=np.uint8)
    return [int(v) for v in np.flatnonzero((alert == 1) & (np.r_[0, alert[:-1]] == 0))]


def alert_episode_end(alert: np.ndarray, start: int) -> int:
    i = start
    while i + 1 < len(alert) and alert[i + 1] == 1:
        i += 1
    return i


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen pipeline-002 policy on a fresh replication cohort.")
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline_v3_work"))
    args = parser.parse_args()

    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import ExtraTreesRegressor

    out = args.out_dir
    public = out / "public"
    dev_ids, dev_x, dev_y, features = _load_bundle(out / "dev_bundle.npz")
    hold_ids, hold_x, _, hold_features = _load_bundle(out / "holdout_inputs.npz")
    if dev_y is None or features != hold_features:
        raise RuntimeError("Development labels missing or feature schema mismatch")

    leakdb_dir = Path(__file__).resolve().parent / "leakdb"
    detector_train_ids: set[int] = set(range(1, 21))
    for number in range(1, 6):
        ids = challenge_ids(leakdb_dir, number)
        detector_train_ids.update(ids[1::2])
    missing = sorted(detector_train_ids.difference(dev_ids))
    if missing:
        raise RuntimeError(f"Detector training scenarios missing: {missing}")

    detector = fit_model(dev_x, dev_y, sorted(detector_train_ids))

    topology = json.loads((public / "topology.json").read_text(encoding="utf-8"))
    coords = topology["coordinates"]
    candidates = topology["candidate_nodes"]
    localizer_events = json.loads((out / "localizer_dev_labels.json").read_text(encoding="utf-8"))
    localizer_events = [e for e in localizer_events if e["node_id"] in coords]
    loc_x = np.stack([event_signature(dev_x[e["scenario_id"]], e["start_idx"]) for e in localizer_events])
    loc_y = np.asarray([coords[e["node_id"]] for e in localizer_events], dtype=np.float64)
    localizer = ExtraTreesRegressor(
        n_estimators=800,
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
        trigger_prob = causal_smooth(raw, ALERT_SMOOTH)
        trigger = (trigger_prob >= ALERT_THRESHOLD).astype(np.uint8)
        alert = latch_alert(trigger, ALERT_HOLD)
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
        "algorithm": "Pay Your Way Net1 pipeline v3 frozen replication",
        "detector": {
            "family": "identical pipeline-002 HGB model/training split",
            "training_scenario_ids": sorted(detector_train_ids),
            "alert_threshold": ALERT_THRESHOLD,
            "alert_smooth_window": ALERT_SMOOTH,
            "alert_hold_steps": ALERT_HOLD,
            "note": "No calibration or tuning uses pipeline-002 or pipeline-003 holdout outcomes. Policy copied verbatim from frozen pipeline 002."
        },
        "localizer": {
            "family": "identical pipeline-001/002 ExtraTrees coordinate localizer v1",
            "training_event_count": len(localizer_events),
            "observation_steps_after_alert": LOCALIZATION_OBSERVATION_STEPS,
            "observation_hours_after_alert": LOCALIZATION_OBSERVATION_STEPS * 0.5,
            "candidate_node_count": len(candidates)
        },
        "holdout_scenario_count": len(hold_ids),
        "raw_sensor_count": len(features or []),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No pipeline-003 holdout leak timing or node labels are opened by prediction."
    }
    (public / "pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen pipeline replicated on {len(hold_ids)} sealed scenarios with {sum(len(r['alert_episodes']) for r in results)} episodes")


if __name__ == "__main__":
    main()
