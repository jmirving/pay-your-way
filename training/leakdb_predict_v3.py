from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from leakdb_sealed import _balanced_rows, _binary_metrics, _load_bundle, _require_numpy
from leakdb_predict_v2 import causal_smooth, scenario_features


def fit_model(xs, ys, ids, random_state: int = 0):
    np = _require_numpy()
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(random_state)
    chunks_x, chunks_y = [], []
    for sid in ids:
        x_bal, y_bal = _balanced_rows(
            scenario_features(xs[sid]), ys[sid], rng, max_normal_ratio=5.0
        )
        chunks_x.append(x_bal)
        chunks_y.append(y_bal)

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.055,
        max_depth=7,
        l2_regularization=0.6,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(np.concatenate(chunks_x), np.concatenate(chunks_y))
    return model


def raw_probabilities(model, x):
    return model.predict_proba(scenario_features(x))[:, 1]


def event_onsets(y):
    np = _require_numpy()
    y = np.asarray(y, dtype=np.uint8)
    return np.flatnonzero((y == 1) & (np.r_[0, y[:-1]] == 0))


def any_event_recall(y, pred, tolerance_steps: int = 10) -> tuple[int, int]:
    np = _require_numpy()
    y = np.asarray(y, dtype=np.uint8)
    pred = np.asarray(pred, dtype=np.uint8)
    onsets = event_onsets(y)
    hits = 0
    for onset in onsets:
        if np.any(pred[onset : onset + tolerance_steps] == 1):
            hits += 1
    return hits, int(len(onsets))


def official_early_score(y, alert, tolerance_steps: int = 10, density_threshold: float = 0.75) -> tuple[float, int]:
    """Match LeakDB's early-detection scoring shape for one scenario."""
    np = _require_numpy()
    y = np.asarray(y, dtype=np.uint8)
    alert = np.asarray(alert, dtype=np.uint8)
    scores: list[float] = []
    for onset in event_onsets(y):
        window = alert[onset : onset + tolerance_steps]
        if len(window) and np.any(window == 1) and float(np.mean(window)) > density_threshold:
            first = int(np.flatnonzero(window == 1)[0]) + 1
            scores.append(2.0 / (1.0 + math.exp((5.0 / tolerance_steps) * first)))
        else:
            scores.append(0.0)
    return float(sum(scores)), len(scores)


def latch_alert(trigger, hold_steps: int):
    np = _require_numpy()
    trigger = np.asarray(trigger, dtype=np.uint8)
    out = np.zeros_like(trigger)
    hold_until = -1
    for i, active in enumerate(trigger):
        if active:
            hold_until = max(hold_until, i + hold_steps - 1)
        if i <= hold_until:
            out[i] = 1
    return out


def alert_episodes(alert) -> int:
    np = _require_numpy()
    alert = np.asarray(alert, dtype=np.uint8)
    return int(np.sum((alert == 1) & (np.r_[0, alert[:-1]] == 0)))


def state_validation(xs, ys, ids, raw_probs, smooth_window: int, threshold: float):
    np = _require_numpy()
    all_true, all_pred = [], []
    event_hits = event_total = 0
    no_leak_positive = no_leak_steps = 0
    for sid in ids:
        y = np.asarray(ys[sid], dtype=np.uint8)
        probs = causal_smooth(raw_probs[sid], smooth_window)
        pred = (probs >= threshold).astype(np.uint8)
        all_true.append(y)
        all_pred.append(pred)
        hits, total = any_event_recall(y, pred)
        event_hits += hits
        event_total += total
        if not np.any(y):
            no_leak_positive += int(np.sum(pred))
            no_leak_steps += len(pred)
    metrics = _binary_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    event_recall = event_hits / event_total if event_total else 0.0
    no_leak_fp = no_leak_positive / no_leak_steps if no_leak_steps else 0.0
    objective = metrics["f1"] + 0.15 * event_recall - 0.5 * no_leak_fp
    return objective, metrics, event_recall, no_leak_fp


def alert_validation(xs, ys, ids, raw_probs, trigger_smooth: int, threshold: float, hold_steps: int):
    np = _require_numpy()
    early_sum = 0.0
    event_total = 0
    no_leak_positive = no_leak_steps = no_leak_episodes = 0
    no_leak_days = 0.0
    all_true, all_alert = [], []

    for sid in ids:
        y = np.asarray(ys[sid], dtype=np.uint8)
        trigger_prob = causal_smooth(raw_probs[sid], trigger_smooth)
        trigger = (trigger_prob >= threshold).astype(np.uint8)
        alert = latch_alert(trigger, hold_steps)
        score, events = official_early_score(y, alert)
        early_sum += score
        event_total += events
        all_true.append(y)
        all_alert.append(alert)
        if not np.any(y):
            no_leak_positive += int(np.sum(alert))
            no_leak_steps += len(alert)
            no_leak_episodes += alert_episodes(alert)
            no_leak_days += len(alert) / 48.0

    early = early_sum / event_total if event_total else 0.0
    no_leak_fp = no_leak_positive / no_leak_steps if no_leak_steps else 0.0
    episodes_per_30d = (no_leak_episodes / no_leak_days * 30.0) if no_leak_days else 0.0
    step_metrics = _binary_metrics(np.concatenate(all_true), np.concatenate(all_alert))
    feasible = no_leak_fp <= 0.02 and episodes_per_30d <= 3.0
    fallback_objective = early - 2.0 * no_leak_fp - 0.02 * episodes_per_30d
    return {
        "early_score": early,
        "events": event_total,
        "no_leak_positive_fraction": no_leak_fp,
        "no_leak_alert_episodes_per_30d": episodes_per_30d,
        "step_metrics": step_metrics,
        "feasible": feasible,
        "fallback_objective": fallback_objective,
    }


def previous_challenge_ids(leakdb_dir: Path, challenge_number: int) -> list[int]:
    path = leakdb_dir / f"challenge_{challenge_number:03d}_prediction_freeze.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(v) for v in payload["holdout_scenario_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train state and operator-alert models for sealed LeakDB challenge 003.")
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
    validation_set = set(previous_challenge_ids(leakdb_dir, 2))
    val_ids = [sid for sid in dev_ids if sid in validation_set]
    train_ids = [sid for sid in dev_ids if sid not in validation_set]
    if len(val_ids) < 20:
        raise RuntimeError("Challenge 002 burned scenarios are required as validation data")

    model = fit_model(dev_x, dev_y, train_ids)
    val_raw = {sid: raw_probabilities(model, dev_x[sid]) for sid in val_ids}

    best_state = None
    for smooth_window in (1, 3, 6):
        for threshold in np.linspace(0.35, 0.90, 23):
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
    for trigger_smooth in (1, 2, 3):
        for threshold in np.linspace(0.30, 0.90, 25):
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

    # Hyperparameters are now frozen from burned validation data. Refit the model on all burned scenarios.
    model = fit_model(dev_x, dev_y, dev_ids)
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
        "algorithm": "LeakDB-Net1 dual-output local-calibrated HGB v3",
        "purpose": {
            "state": "estimate whether leakage is active at each timestep",
            "alert": "produce a sustained operator-facing alert soon after likely leak onset",
        },
        "development_scenarios": dev_ids,
        "training_scenarios_for_hyperparameter_search": train_ids,
        "validation_scenarios": val_ids,
        "final_model_training_scenarios": dev_ids,
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
        "label_boundary": "No challenge-003 holdout label file is opened by this prediction process.",
    }
    (public / "prediction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "v3 predicted "
        f"{len(hold_ids)} sealed scenarios; state threshold={state_threshold:.3f}/smooth={state_smooth}; "
        f"alert threshold={alert_threshold:.3f}/smooth={alert_trigger_smooth}/hold={alert_hold_steps}"
    )


if __name__ == "__main__":
    main()
