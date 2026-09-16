from __future__ import annotations

import argparse
import json
from pathlib import Path

from leakdb_sealed import _balanced_rows, _binary_metrics, _event_metrics, _load_bundle, _require_numpy


def scenario_features(x, calibration_steps: int = 96, period: int = 48):
    """Label-free, scenario-local features to reduce cross-scenario hydraulic shift."""
    np = _require_numpy()
    x = np.asarray(x, dtype=np.float64)
    n_cal = min(calibration_steps, len(x))
    if n_cal < period:
        raise ValueError("scenario is too short for local calibration")
    cal = x[:n_cal]
    center = np.nanmedian(cal, axis=0)
    mad = np.nanmedian(np.abs(cal - center), axis=0)
    scale = 1.4826 * mad
    std = np.nanstd(cal, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))

    local_z = np.clip(np.nan_to_num((x - center) / scale, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)

    phase_center = np.empty((period, x.shape[1]), dtype=np.float64)
    for phase in range(period):
        rows = cal[phase:n_cal:period]
        phase_center[phase] = np.nanmedian(rows, axis=0) if len(rows) else center
    expected = phase_center[np.arange(len(x)) % period]
    phase_resid = np.clip(
        np.nan_to_num((x - expected) / scale, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20
    )

    d1 = np.zeros_like(phase_resid)
    d1[1:] = phase_resid[1:] - phase_resid[:-1]
    seasonal = np.zeros_like(phase_resid)
    if len(x) > period:
        seasonal[period:] = (x[period:] - x[:-period]) / scale
        seasonal = np.clip(np.nan_to_num(seasonal, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)

    return np.concatenate([local_z, phase_resid, d1, seasonal], axis=1).astype(np.float32)


def causal_smooth(probs, window: int):
    np = _require_numpy()
    probs = np.asarray(probs, dtype=np.float64)
    if window <= 1:
        return probs
    c = np.cumsum(np.r_[0.0, probs])
    n = np.arange(1, len(probs) + 1)
    starts = np.maximum(0, n - window)
    return (c[n] - c[starts]) / (n - starts)


def fit_model(xs, ys, ids, random_state: int = 0):
    np = _require_numpy()
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(random_state)
    chunks_x, chunks_y = [], []
    for sid in ids:
        x_bal, y_bal = _balanced_rows(scenario_features(xs[sid]), ys[sid], rng, max_normal_ratio=5.0)
        chunks_x.append(x_bal)
        chunks_y.append(y_bal)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.06,
        max_depth=7,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(np.concatenate(chunks_x), np.concatenate(chunks_y))
    return model


def probabilities(model, x, smooth_window: int):
    raw = model.predict_proba(scenario_features(x))[:, 1]
    return causal_smooth(raw, smooth_window)


def validation_objective(xs, ys, ids, model, smooth_window: int, threshold: float):
    np = _require_numpy()
    all_true, all_pred, event_recalls = [], [], []
    no_leak_positive = no_leak_steps = 0
    for sid in ids:
        y = np.asarray(ys[sid], dtype=np.uint8)
        pred = (probabilities(model, xs[sid], smooth_window) >= threshold).astype(np.uint8)
        all_true.append(y)
        all_pred.append(pred)
        event_recalls.append(_event_metrics(y, pred)["event_recall"])
        if not np.any(y):
            no_leak_positive += int(np.sum(pred))
            no_leak_steps += len(pred)
    metrics = _binary_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    event_recall = float(np.mean(event_recalls)) if event_recalls else 0.0
    no_leak_fp = no_leak_positive / no_leak_steps if no_leak_steps else 0.0
    objective = metrics["f1"] + 0.25 * event_recall - 0.75 * no_leak_fp
    return objective, metrics, event_recall, no_leak_fp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("challenge_work"))
    args = parser.parse_args()

    np = _require_numpy()
    from sklearn import __version__ as sklearn_version

    public = args.out_dir / "public"
    dev_ids, dev_x, dev_y, features = _load_bundle(args.out_dir / "dev_bundle.npz")
    hold_ids, hold_x, _, hold_features = _load_bundle(args.out_dir / "holdout_inputs.npz")
    if dev_y is None or features != hold_features:
        raise RuntimeError("Development labels missing or feature schema mismatch")

    preferred_validation = {163, 209, 245, 430, 437, 706, 726, 772, 837}
    val_ids = [sid for sid in dev_ids if sid in preferred_validation]
    train_ids = [sid for sid in dev_ids if sid not in set(val_ids)]
    if len(val_ids) < 5:
        raise RuntimeError("Expected challenge-001 hard cases in development set")

    model = fit_model(dev_x, dev_y, train_ids)
    best = None
    for smooth_window in (1, 3, 6):
        for threshold in np.linspace(0.50, 0.97, 20):
            obj, metrics, event_recall, no_leak_fp = validation_objective(
                dev_x, dev_y, val_ids, model, smooth_window, float(threshold)
            )
            candidate = (obj, metrics["f1"], event_recall, -no_leak_fp, smooth_window, float(threshold), metrics, no_leak_fp)
            if best is None or candidate[:4] > best[:4]:
                best = candidate
    assert best is not None
    smooth_window, threshold, val_metrics, no_leak_fp = best[4], best[5], best[6], best[7]

    model = fit_model(dev_x, dev_y, dev_ids)
    payload: dict[str, object] = {"scenario_ids": np.asarray(hold_ids, dtype=np.int32)}
    for sid in hold_ids:
        probs = probabilities(model, hold_x[sid], smooth_window)
        payload[f"prob_{sid}"] = probs.astype(np.float32)
        payload[f"pred_{sid}"] = (probs >= threshold).astype(np.uint8)
    np.savez_compressed(public / "predictions.npz", **payload)

    report = {
        "algorithm": "LeakDB-Net1 local-calibrated HGB v2",
        "development_scenarios": dev_ids,
        "training_scenarios": train_ids,
        "validation_scenarios": val_ids,
        "calibration_steps": 96,
        "seasonal_period": 48,
        "smooth_window": smooth_window,
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "validation_no_leak_positive_fraction": no_leak_fp,
        "feature_count_raw": len(features or []),
        "feature_count_engineered": int(next(iter(dev_x.values())).shape[1] * 4),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No holdout label file is opened by this prediction process.",
    }
    (public / "prediction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"v2 predicted {len(hold_ids)} sealed scenarios; threshold={threshold:.3f}, smooth={smooth_window}")


if __name__ == "__main__":
    main()
