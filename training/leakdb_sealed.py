from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Iterable


def _require_numpy():
    import numpy as np
    return np


def _save_bundle(path: Path, scenario_ids: list[int], xs: dict[int, object], ys: dict[int, object] | None = None, features: list[str] | None = None) -> None:
    np = _require_numpy()
    payload: dict[str, object] = {"scenario_ids": np.asarray(scenario_ids, dtype=np.int32)}
    if features is not None:
        payload["features"] = np.asarray(features, dtype=str)
    for sid in scenario_ids:
        payload[f"X_{sid}"] = np.asarray(xs[sid], dtype=np.float32)
        if ys is not None:
            payload[f"y_{sid}"] = np.asarray(ys[sid], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_bundle(path: Path) -> tuple[list[int], dict[int, object], dict[int, object] | None, list[str] | None]:
    np = _require_numpy()
    data = np.load(path, allow_pickle=False)
    ids = [int(v) for v in data["scenario_ids"]]
    xs = {sid: data[f"X_{sid}"] for sid in ids if f"X_{sid}" in data}
    ys = {sid: data[f"y_{sid}"] for sid in ids if f"y_{sid}" in data}
    if not ys:
        ys = None
    features = [str(v) for v in data["features"]] if "features" in data else None
    return ids, xs, ys, features


def _load_leakdb(ids: list[int], cache_dir: Path):
    from water_benchmark_hub import load
    benchmark = load("KIOS-LeakDB")
    result = benchmark.load_data(ids, use_net1=True, download_dir=str(cache_dir), return_X_y=True, return_features_desc=True, return_leak_locations=False, verbose=False)
    features = [str(v) for v in result["features_desc"]]
    xs: dict[int, object] = {}
    ys: dict[int, object] = {}
    for sid in ids:
        x, y = result[sid]
        xs[sid] = x
        ys[sid] = y
    return xs, ys, features


def prepare(args: argparse.Namespace) -> None:
    np = _require_numpy()
    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    dev_ids = list(range(1, args.dev_count + 1))
    candidates = [sid for sid in range(args.holdout_min, args.holdout_max + 1) if sid not in dev_ids]
    holdout_ids = sorted(secrets.SystemRandom().sample(candidates, args.holdout_count))
    print(f"Preparing {len(dev_ids)} development and {len(holdout_ids)} sealed holdout scenarios.")
    dev_x, dev_y, features = _load_leakdb(dev_ids, cache)
    hold_x, hold_y, hold_features = _load_leakdb(holdout_ids, cache)
    if hold_features != features:
        raise RuntimeError("Development and holdout feature schemas differ")
    _save_bundle(out / "dev_bundle.npz", dev_ids, dev_x, dev_y, features)
    _save_bundle(out / "holdout_inputs.npz", holdout_ids, hold_x, None, features)
    label_payload: dict[str, object] = {"scenario_ids": np.asarray(holdout_ids, dtype=np.int32)}
    for sid in holdout_ids:
        label_payload[f"y_{sid}"] = np.asarray(hold_y[sid], dtype=np.uint8)
    np.savez_compressed(secret / "labels.npz", **label_payload)
    manifest = {"benchmark":"KIOS-LeakDB","network":"Net1","development_scenario_ids":dev_ids,"holdout_scenario_ids":holdout_ids,"holdout_selection":"cryptographically random sample at workflow runtime","holdout_range":[args.holdout_min,args.holdout_max],"label_boundary":"holdout labels stored only in sealed-leakdb-labels artifact","feature_count":len(features)}
    (public / "challenge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _normal_baseline(xs: dict[int, object], ys: dict[int, object], ids: Iterable[int]):
    np = _require_numpy()
    normal_rows = []
    for sid in ids:
        x, y = np.asarray(xs[sid], dtype=np.float64), np.asarray(ys[sid], dtype=np.uint8)
        normal = x[y == 0]
        if len(normal): normal_rows.append(normal)
    if not normal_rows: raise RuntimeError("No normal development samples available")
    rows = np.concatenate(normal_rows, axis=0)
    center = np.nanmedian(rows, axis=0)
    mad = np.nanmedian(np.abs(rows - center), axis=0)
    scale = 1.4826 * mad
    std = np.nanstd(rows, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    return center, scale


def _engineer(x, center, scale, seasonal_period: int = 48):
    np = _require_numpy()
    z = np.nan_to_num((np.asarray(x, dtype=np.float64) - center) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    z = np.clip(z, -20.0, 20.0)
    d1 = np.zeros_like(z); d1[1:] = z[1:] - z[:-1]
    seasonal = np.zeros_like(z)
    if len(z) > seasonal_period: seasonal[seasonal_period:] = z[seasonal_period:] - z[:-seasonal_period]
    return np.concatenate([z, d1, seasonal], axis=1).astype(np.float32)


def _balanced_rows(features, labels, rng, max_normal_ratio: float = 4.0):
    np = _require_numpy(); labels = np.asarray(labels, dtype=np.uint8)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    take_neg = min(len(neg), 5000 if len(pos) == 0 else max(int(len(pos) * max_normal_ratio), 1000))
    if len(neg) > take_neg: neg = rng.choice(neg, size=take_neg, replace=False)
    idx = np.concatenate([pos, neg]); rng.shuffle(idx)
    return features[idx], labels[idx]


def _smooth(probs, window: int = 3):
    np = _require_numpy(); probs = np.asarray(probs, dtype=np.float64)
    if window <= 1 or len(probs) < window: return probs
    return np.convolve(probs, np.ones(window) / window, mode="same")


def _binary_metrics(y_true, y_pred) -> dict[str, float]:
    np = _require_numpy(); y_true = np.asarray(y_true, dtype=np.uint8); y_pred = np.asarray(y_pred, dtype=np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1))); tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1))); fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0; tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"precision":precision,"recall":recall,"f1":f1,"true_negative_rate":tnr}


def _event_metrics(y_true, y_pred, tolerance_steps: int = 10) -> dict[str, float | int | None]:
    np = _require_numpy(); y_true = np.asarray(y_true, dtype=np.uint8); y_pred = np.asarray(y_pred, dtype=np.uint8)
    transitions = np.flatnonzero((y_true == 1) & (np.r_[0, y_true[:-1]] == 0)); delays: list[int] = []; hits = 0
    for onset in transitions:
        hit = np.flatnonzero(y_pred[onset:onset + tolerance_steps] == 1)
        if len(hit): hits += 1; delays.append(int(hit[0]))
    return {"events":int(len(transitions)),"events_detected_within_tolerance":int(hits),"event_recall":hits/len(transitions) if len(transitions) else 0.0,"median_detection_delay_steps":float(np.median(delays)) if delays else None}


def _fit_model(xs, ys, train_ids, center, scale, random_state: int = 0):
    np = _require_numpy(); from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(random_state); x_chunks, y_chunks = [], []
    for sid in train_ids:
        x_bal, y_bal = _balanced_rows(_engineer(xs[sid], center, scale), ys[sid], rng); x_chunks.append(x_bal); y_chunks.append(y_bal)
    model = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_depth=6, l2_regularization=0.25, class_weight="balanced", random_state=random_state)
    model.fit(np.concatenate(x_chunks), np.concatenate(y_chunks)); return model


def _predict_scenario(model, x, center, scale):
    return _smooth(model.predict_proba(_engineer(x, center, scale))[:, 1], window=3)


def predict(args: argparse.Namespace) -> None:
    np = _require_numpy(); from sklearn import __version__ as sklearn_version
    out, public = args.out_dir, args.out_dir / "public"
    dev_ids, dev_x, dev_y, features = _load_bundle(out / "dev_bundle.npz"); hold_ids, hold_x, _, hold_features = _load_bundle(out / "holdout_inputs.npz")
    if dev_y is None: raise RuntimeError("Development bundle is missing labels")
    if features != hold_features: raise RuntimeError("Feature schema mismatch")
    split = max(1, int(round(len(dev_ids) * 0.7))); train_ids, val_ids = dev_ids[:split], dev_ids[split:]
    if not val_ids: val_ids, train_ids = dev_ids[-1:], dev_ids[:-1]
    center, scale = _normal_baseline(dev_x, dev_y, train_ids); model = _fit_model(dev_x, dev_y, train_ids, center, scale)
    val_probs = {sid:_predict_scenario(model, dev_x[sid], center, scale) for sid in val_ids}; best = None
    for threshold in np.linspace(0.10, 0.90, 17):
        all_true, all_pred, event_hits = [], [], []
        for sid in val_ids:
            y = np.asarray(dev_y[sid], dtype=np.uint8); p = (val_probs[sid] >= threshold).astype(np.uint8)
            all_true.append(y); all_pred.append(p); event_hits.append(_event_metrics(y, p)["event_recall"])
        metrics = _binary_metrics(np.concatenate(all_true), np.concatenate(all_pred)); mean_event = float(np.mean(event_hits)) if event_hits else 0.0
        candidate = (metrics["f1"] + 0.25 * mean_event, metrics["f1"], mean_event, float(threshold), metrics)
        if best is None or candidate[:3] > best[:3]: best = candidate
    assert best is not None
    threshold, val_metrics = best[3], best[4]
    center, scale = _normal_baseline(dev_x, dev_y, dev_ids); model = _fit_model(dev_x, dev_y, dev_ids, center, scale)
    payload: dict[str, object] = {"scenario_ids":np.asarray(hold_ids, dtype=np.int32)}
    for sid in hold_ids:
        probs = _predict_scenario(model, hold_x[sid], center, scale); payload[f"prob_{sid}"] = probs.astype(np.float32); payload[f"pred_{sid}"] = (probs >= threshold).astype(np.uint8)
    np.savez_compressed(public / "predictions.npz", **payload)
    report = {"algorithm":"LeakDB-Net1 HGB v1","development_scenarios":dev_ids,"training_scenarios":train_ids,"validation_scenarios":val_ids,"sealed_holdout_scenarios":hold_ids,"threshold":threshold,"validation_metrics":val_metrics,"feature_count_raw":len(features or []),"feature_count_engineered":int(next(iter(dev_x.values())).shape[1]*3),"scikit_learn_version":sklearn_version,"label_boundary":"No holdout label file is opened by predict()."}
    (public / "prediction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen inference produced predictions for {len(hold_ids)} sealed scenarios; threshold={threshold:.2f}")


def score(args: argparse.Namespace) -> None:
    np = _require_numpy(); pred = np.load(args.predictions, allow_pickle=False); labels = np.load(args.labels, allow_pickle=False)
    pred_ids, label_ids = [int(v) for v in pred["scenario_ids"]], [int(v) for v in labels["scenario_ids"]]
    if pred_ids != label_ids: raise RuntimeError("Prediction/label scenario IDs differ")
    all_true, all_pred, rows = [], [], []
    for sid in pred_ids:
        y, p = labels[f"y_{sid}"].astype(np.uint8), pred[f"pred_{sid}"].astype(np.uint8); metrics = _binary_metrics(y,p); events = _event_metrics(y,p)
        rows.append({"scenario_id":sid,**metrics,**events}); all_true.append(y); all_pred.append(p)
    aggregate = _binary_metrics(np.concatenate(all_true), np.concatenate(all_pred)); total_events = sum(r["events"] for r in rows); hits = sum(r["events_detected_within_tolerance"] for r in rows)
    result = {"scenario_count":len(pred_ids),"aggregate":aggregate,"event_recall_within_10_steps":hits/total_events if total_events else 0.0,"events":total_events,"events_detected":hits,"per_scenario":rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result["aggregate"], indent=2)); print(f"event recall within 10 steps: {result['event_recall_within_10_steps']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare, predict, and score a sealed LeakDB benchmark."); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out-dir",type=Path,default=Path("challenge_work")); p.add_argument("--dev-count",type=int,default=20); p.add_argument("--holdout-count",type=int,default=20); p.add_argument("--holdout-min",type=int,default=101); p.add_argument("--holdout-max",type=int,default=1000); p.set_defaults(func=prepare)
    p = sub.add_parser("predict"); p.add_argument("--out-dir",type=Path,default=Path("challenge_work")); p.set_defaults(func=predict)
    p = sub.add_parser("score"); p.add_argument("--predictions",type=Path,required=True); p.add_argument("--labels",type=Path,required=True); p.add_argument("--output",type=Path,default=Path("sealed_score.json")); p.set_defaults(func=score)
    return parser


def main() -> None:
    args = build_parser().parse_args(); args.func(args)


if __name__ == "__main__": main()
