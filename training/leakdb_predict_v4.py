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

# Challenge 003 is now burned. Keep a fixed, deliberately diverse calibration set:
# four no-leak scenarios, hard low-recall leaks, and several easier leaks.
VALIDATION_IDS = {
    128, 139, 142, 250, 314, 365, 373, 442, 503, 523,
    581, 651, 654, 715, 746, 758, 801, 912, 967, 996,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and freeze calibration-stable state/alert outputs for LeakDB challenge 004.")
    parser.add_argument("--out-dir", type=Path, default=Path("challenge_work"))
    args = parser.parse_args()

    np = _require_numpy()
    from sklearn import __version__ as sklearn_version

    public = args.out_dir / "public"
    dev_ids, dev_x, dev_y, features = _load_bundle(args.out_dir / "dev_bundle.npz")
    hold_ids, hold_x, _, hold_features = _load_bundle(args.out_dir / "holdout_inputs.npz")
    if dev_y is None or features != hold_features:
        raise RuntimeError("Development labels missing or feature schema mismatch")

    val_ids = [sid for sid in dev_ids if sid in VALIDATION_IDS]
    train_ids = [sid for sid in dev_ids if sid not in VALIDATION_IDS]
    if set(val_ids) != VALIDATION_IDS:
        missing = sorted(VALIDATION_IDS.difference(val_ids))
        raise RuntimeError(f"Challenge 004 calibration scenarios missing from development bundle: {missing}")

    # Important difference from v3: this exact fitted model is the one used on the
    # sealed holdout. We do not refit after threshold/policy calibration.
    model = fit_model(dev_x, dev_y, train_ids)
    val_raw = {sid: raw_probabilities(model, dev_x[sid]) for sid in val_ids}

    best_state = None
    for smooth_window in (1, 3, 6, 12):
        for threshold in np.linspace(0.20, 0.90, 29):
            obj, metrics, event_recall, no_leak_fp = state_validation(
                dev_x, dev_y, val_ids, val_raw, smooth_window, float(threshold)
            )
            candidate = (
                obj,
                metrics["f1"],
                event_recall,
                -no_leak_fp,
                smooth_window,
                float(threshold),
                metrics,
                no_leak_fp,
            )
            if best_state is None or candidate[:4] > best_state[:4]:
                best_state = candidate
    assert best_state is not None
    state_smooth = best_state[4]
    state_threshold = best_state[5]
    state_val_metrics = best_state[6]
    state_no_leak_fp = best_state[7]

    best_alert = None
    for trigger_smooth in (1, 2, 3, 6):
        for threshold in np.linspace(0.25, 0.90, 27):
            for hold_steps in (8, 10, 12):
                result = alert_validation(
                    dev_x,
                    dev_y,
                    val_ids,
                    val_raw,
                    trigger_smooth,
                    float(threshold),
                    hold_steps,
                )
                key = (
                    1 if result["feasible"] else 0,
                    result["early_score"] if result["feasible"] else result["fallback_objective"],
                    -result["no_leak_positive_fraction"],
                    -result["no_leak_alert_episodes_per_30d"],
                )
                candidate = (key, trigger_smooth, float(threshold), hold_steps, result)
                if best_alert is None or candidate[0] > best_alert[0]:
                    best_alert = candidate
    assert best_alert is not None
    alert_trigger_smooth = best_alert[1]
    alert_threshold = best_alert[2]
    alert_hold_steps = best_alert[3]
    alert_val = best_alert[4]

    payload: dict[str, object] = {"scenario_ids": np.asarray(hold_ids, dtype=np.int32)}
    for sid in hold_ids:
        raw = raw_probabilities(model, hold_x[sid])
        state_prob = causal_smooth(raw, state_smooth)
        state_pred = (state_prob >= state_threshold).astype(np.uint8)
        trigger_prob = causal_smooth(raw, alert_trigger_smooth)
        trigger = (trigger_prob >= alert_threshold).astype(np.uint8)
        alert_pred = latch_alert(trigger, alert_hold_steps)

        payload[f"prob_{sid}"] = raw.astype(np.float32)
        payload[f"state_prob_{sid}"] = state_prob.astype(np.float32)
        payload[f"pred_{sid}"] = state_pred
        payload[f"state_pred_{sid}"] = state_pred
        payload[f"alert_pred_{sid}"] = alert_pred

    np.savez_compressed(public / "predictions.npz", **payload)

    report = {
        "algorithm": "LeakDB-Net1 calibration-stable dual-output HGB v4",
        "method_change": "Model is fitted once on training_scenarios, calibrated on disjoint burned validation_scenarios, and not refit after calibration.",
        "development_scenarios": dev_ids,
        "training_scenarios": train_ids,
        "validation_scenarios": val_ids,
        "sealed_holdout_scenarios": hold_ids,
        "state": {
            "smooth_window": state_smooth,
            "threshold": state_threshold,
            "validation_metrics": state_val_metrics,
            "validation_no_leak_positive_fraction": state_no_leak_fp,
        },
        "alert": {
            "trigger_smooth_window": alert_trigger_smooth,
            "trigger_threshold": alert_threshold,
            "hold_steps": alert_hold_steps,
            "validation": alert_val,
        },
        "feature_count_raw": len(features or []),
        "feature_count_engineered": int(next(iter(dev_x.values())).shape[1] * 4),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No challenge-004 holdout label file is opened by this prediction process.",
    }
    (public / "prediction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"v4 predicted {len(hold_ids)} sealed scenarios; "
        f"state={state_threshold:.3f}/smooth{state_smooth}; "
        f"alert={alert_threshold:.3f}/smooth{alert_trigger_smooth}/hold{alert_hold_steps}"
    )


if __name__ == "__main__":
    main()
