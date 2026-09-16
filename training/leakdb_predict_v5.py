from __future__ import annotations

import argparse
import json
from pathlib import Path

from leakdb_sealed import _load_bundle, _require_numpy
from leakdb_predict_v3 import (
    alert_validation,
    causal_smooth,
    fit_model,
    latch_alert,
    raw_probabilities,
    state_validation,
)


def challenge_ids(leakdb_dir: Path, number: int) -> list[int]:
    payload = json.loads(
        (leakdb_dir / f"challenge_{number:03d}_prediction_freeze.json").read_text(encoding="utf-8")
    )
    return [int(v) for v in payload["holdout_scenario_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-cohort robust calibration for sealed LeakDB challenge 005.")
    parser.add_argument("--out-dir", type=Path, default=Path("challenge_work"))
    args = parser.parse_args()

    np = _require_numpy()
    from sklearn import __version__ as sklearn_version

    public = args.out_dir / "public"
    dev_ids, dev_x, dev_y, features = _load_bundle(args.out_dir / "dev_bundle.npz")
    hold_ids, hold_x, _, hold_features = _load_bundle(args.out_dir / "holdout_inputs.npz")
    if dev_y is None or features != hold_features:
        raise RuntimeError("Development labels missing or feature schema mismatch")

    leakdb_dir = Path(__file__).resolve().parent / "leakdb"
    groups: dict[str, list[int]] = {}
    validation_ids: set[int] = set()
    for number in range(1, 5):
        ids = sorted(challenge_ids(leakdb_dir, number))
        val = ids[::2]
        groups[f"challenge_{number:03d}"] = val
        validation_ids.update(val)

    val_ids = [sid for sid in dev_ids if sid in validation_ids]
    train_ids = [sid for sid in dev_ids if sid not in validation_ids]
    if len(val_ids) != sum(len(v) for v in groups.values()):
        raise RuntimeError("Expected validation scenarios from every burned challenge")

    # One fitted model. Calibration never changes its parameters, and there is no
    # post-calibration refit. This keeps probabilities stable between validation and holdout.
    model = fit_model(dev_x, dev_y, train_ids)
    val_raw = {sid: raw_probabilities(model, dev_x[sid]) for sid in val_ids}

    best_state = None
    for smooth_window in (1, 3, 6, 12):
        for threshold in np.linspace(0.20, 0.95, 31):
            group_results = {}
            for name, ids in groups.items():
                obj, metrics, event_recall, no_leak_fp = state_validation(
                    dev_x, dev_y, ids, val_raw, smooth_window, float(threshold)
                )
                group_results[name] = {
                    "objective": obj,
                    "metrics": metrics,
                    "event_recall": event_recall,
                    "no_leak_positive_fraction": no_leak_fp,
                }
            f1s = [r["metrics"]["f1"] for r in group_results.values()]
            event_recalls = [r["event_recall"] for r in group_results.values()]
            no_leak_fps = [r["no_leak_positive_fraction"] for r in group_results.values()]
            feasible = max(no_leak_fps) <= 0.02
            robust = min(f1s) - 0.5 * max(no_leak_fps)
            key = (
                1 if feasible else 0,
                min(f1s) if feasible else robust,
                sum(f1s) / len(f1s),
                min(event_recalls),
                -max(no_leak_fps),
            )
            candidate = (key, smooth_window, float(threshold), group_results)
            if best_state is None or candidate[0] > best_state[0]:
                best_state = candidate
    assert best_state is not None
    state_smooth, state_threshold, state_groups = best_state[1], best_state[2], best_state[3]

    best_alert = None
    for trigger_smooth in (1, 2, 3, 6):
        for threshold in np.linspace(0.25, 0.95, 29):
            for hold_steps in (8, 10, 12):
                group_results = {
                    name: alert_validation(
                        dev_x,
                        dev_y,
                        ids,
                        val_raw,
                        trigger_smooth,
                        float(threshold),
                        hold_steps,
                    )
                    for name, ids in groups.items()
                }
                early = [r["early_score"] for r in group_results.values()]
                fp = [r["no_leak_positive_fraction"] for r in group_results.values()]
                episodes = [r["no_leak_alert_episodes_per_30d"] for r in group_results.values()]
                feasible = max(fp) <= 0.02 and max(episodes) <= 3.0
                robust = min(
                    r["early_score"]
                    - 2.0 * r["no_leak_positive_fraction"]
                    - 0.02 * r["no_leak_alert_episodes_per_30d"]
                    for r in group_results.values()
                )
                key = (
                    1 if feasible else 0,
                    min(early) if feasible else robust,
                    sum(early) / len(early),
                    -max(fp),
                    -max(episodes),
                )
                candidate = (key, trigger_smooth, float(threshold), hold_steps, group_results)
                if best_alert is None or candidate[0] > best_alert[0]:
                    best_alert = candidate
    assert best_alert is not None
    alert_smooth, alert_threshold, alert_hold, alert_groups = (
        best_alert[1], best_alert[2], best_alert[3], best_alert[4]
    )

    payload: dict[str, object] = {"scenario_ids": np.asarray(hold_ids, dtype=np.int32)}
    for sid in hold_ids:
        raw = raw_probabilities(model, hold_x[sid])
        state_prob = causal_smooth(raw, state_smooth)
        state_pred = (state_prob >= state_threshold).astype(np.uint8)
        trigger_prob = causal_smooth(raw, alert_smooth)
        trigger = (trigger_prob >= alert_threshold).astype(np.uint8)
        alert_pred = latch_alert(trigger, alert_hold)
        payload[f"prob_{sid}"] = raw.astype(np.float32)
        payload[f"state_prob_{sid}"] = state_prob.astype(np.float32)
        payload[f"pred_{sid}"] = state_pred
        payload[f"state_pred_{sid}"] = state_pred
        payload[f"alert_pred_{sid}"] = alert_pred
    np.savez_compressed(public / "predictions.npz", **payload)

    report = {
        "algorithm": "LeakDB-Net1 multi-cohort robust-calibrated dual-output HGB v5",
        "method_change": "Calibration uses disjoint slices from challenges 001-004 and optimizes worst-cohort performance with nuisance constraints; model is not refit after calibration.",
        "development_count": len(dev_ids),
        "training_scenarios": train_ids,
        "validation_groups": groups,
        "sealed_holdout_scenarios": hold_ids,
        "state": {
            "smooth_window": state_smooth,
            "threshold": state_threshold,
            "validation_groups": state_groups,
        },
        "alert": {
            "trigger_smooth_window": alert_smooth,
            "trigger_threshold": alert_threshold,
            "hold_steps": alert_hold,
            "validation_groups": alert_groups,
        },
        "feature_count_raw": len(features or []),
        "feature_count_engineered": int(next(iter(dev_x.values())).shape[1] * 4),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No challenge-005 holdout label file is opened by this prediction process.",
    }
    (public / "prediction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"v5 predicted {len(hold_ids)} sealed scenarios; state={state_threshold:.3f}/smooth{state_smooth}; "
        f"alert={alert_threshold:.3f}/smooth{alert_smooth}/hold{alert_hold}"
    )


if __name__ == "__main__":
    main()
