from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakdb_sealed import _load_bundle


def event_signature(x, start_idx: int, horizon: int = 10, period: int = 48, lookback_days: int = 7) -> np.ndarray:
    """Build a causal five-hour leak signature relative to prior same-time-of-day observations."""
    x = np.asarray(x, dtype=np.float64)
    if start_idx < 0 or start_idx >= len(x):
        raise ValueError(f"invalid event start index {start_idx}")
    end = min(len(x), start_idx + horizon)
    current = x[start_idx:end]
    if len(current) == 0:
        raise ValueError("event window is empty")

    baseline_rows = []
    for day in range(1, lookback_days + 1):
        s = start_idx - day * period
        e = s + len(current)
        if s >= 0 and e <= len(x):
            baseline_rows.append(x[s:e])

    pre_start = max(0, start_idx - lookback_days * period)
    pre = x[pre_start:start_idx]
    if baseline_rows:
        stack = np.stack(baseline_rows, axis=0)
        expected = np.nanmedian(stack, axis=0)
    elif len(pre):
        expected = np.repeat(np.nanmedian(pre, axis=0, keepdims=True), len(current), axis=0)
    else:
        expected = np.repeat(np.nanmedian(x[: min(len(x), period)], axis=0, keepdims=True), len(current), axis=0)

    if len(pre):
        center = np.nanmedian(pre, axis=0)
        mad = np.nanmedian(np.abs(pre - center), axis=0)
        scale = 1.4826 * mad
        std = np.nanstd(pre, axis=0)
        scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    else:
        scale = np.ones(x.shape[1], dtype=np.float64)

    residual = np.nan_to_num((current - expected) / scale, nan=0.0, posinf=20.0, neginf=-20.0)
    residual = np.clip(residual, -20.0, 20.0)
    return np.concatenate(
        [
            np.nanmedian(residual, axis=0),
            np.nanmin(residual, axis=0),
            np.nanmax(residual, axis=0),
            residual[-1],
        ]
    ).astype(np.float32)


def rank_nodes(pred_xy: np.ndarray, coordinates: dict[str, list[float]], candidates: list[str]) -> list[dict]:
    rows = []
    for node in candidates:
        xy = np.asarray(coordinates[node], dtype=np.float64)
        dist = float(np.linalg.norm(xy - pred_xy))
        rows.append({"node_id": node, "map_distance": dist})
    rows.sort(key=lambda row: (row["map_distance"], row["node_id"]))
    return rows


def group_cv(dev_inputs, dev_events, topology, random_state: int = 0) -> dict:
    """Leave-one-burned-challenge-group-out diagnostics for the fixed model family."""
    from sklearn.ensemble import ExtraTreesRegressor

    coords = topology["coordinates"]
    candidates = topology["candidate_nodes"]
    groups = sorted({e["source_group"] for e in dev_events if e["source_group"] != "base"})
    rows = []
    for group in groups:
        train_events = [e for e in dev_events if e["source_group"] != group]
        test_events = [e for e in dev_events if e["source_group"] == group]
        if not train_events or not test_events:
            continue
        x_train = np.stack([event_signature(dev_inputs[e["scenario_id"]], e["start_idx"]) for e in train_events])
        y_train = np.asarray([coords[e["node_id"]] for e in train_events], dtype=np.float64)
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.75,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        top1 = top3 = top5 = 0
        rr = []
        for event in test_events:
            sig = event_signature(dev_inputs[event["scenario_id"]], event["start_idx"])
            pred_xy = model.predict(sig.reshape(1, -1))[0]
            ranking = rank_nodes(pred_xy, coords, candidates)
            rank = next(i + 1 for i, row in enumerate(ranking) if row["node_id"] == event["node_id"])
            top1 += rank <= 1
            top3 += rank <= 3
            top5 += rank <= 5
            rr.append(1.0 / rank)
        n = len(test_events)
        rows.append({
            "held_out_group": group,
            "events": n,
            "top1": top1 / n,
            "top3": top3 / n,
            "top5": top5 / n,
            "mrr": float(np.mean(rr)),
        })
    return {
        "groups": rows,
        "worst_top3": min((r["top3"] for r in rows), default=0.0),
        "mean_top3": float(np.mean([r["top3"] for r in rows])) if rows else 0.0,
        "mean_mrr": float(np.mean([r["mrr"] for r in rows])) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit coordinate-regression localization baseline and rank sealed candidate nodes.")
    parser.add_argument("--out-dir", type=Path, default=Path("localization_work"))
    args = parser.parse_args()

    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import ExtraTreesRegressor

    out = args.out_dir
    public = out / "public"
    dev_ids, dev_x, _, features = _load_bundle(out / "dev_inputs.npz")
    hold_ids, hold_x, _, hold_features = _load_bundle(out / "holdout_inputs.npz")
    if features != hold_features:
        raise RuntimeError("Feature schema mismatch")
    dev_events = json.loads((out / "dev_localization_labels.json").read_text(encoding="utf-8"))
    hold_events = json.loads((out / "holdout_events.json").read_text(encoding="utf-8"))
    topology = json.loads((public / "topology.json").read_text(encoding="utf-8"))
    coords = topology["coordinates"]
    candidates = topology["candidate_nodes"]

    usable_dev = [e for e in dev_events if e["node_id"] in coords]
    x_train = np.stack([event_signature(dev_x[e["scenario_id"]], e["start_idx"]) for e in usable_dev])
    y_train = np.asarray([coords[e["node_id"]] for e in usable_dev], dtype=np.float64)

    cv = group_cv(dev_x, usable_dev, topology)
    model = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = []
    for event in hold_events:
        sig = event_signature(hold_x[event["scenario_id"]], event["start_idx"])
        pred_xy = model.predict(sig.reshape(1, -1))[0]
        ranking = rank_nodes(pred_xy, coords, candidates)
        predictions.append({
            "event_id": event["event_id"],
            "scenario_id": event["scenario_id"],
            "start_idx": event["start_idx"],
            "predicted_xy": [float(pred_xy[0]), float(pred_xy[1])],
            "ranked_nodes": ranking,
        })

    (public / "localization_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    report = {
        "algorithm": "LeakDB-Net1 ExtraTrees coordinate localizer v1",
        "task": "conditional localization given true leak-event onset; node ID withheld",
        "feature_window_steps": 10,
        "feature_window_hours": 5,
        "lookback_days": 7,
        "signature": "per-sensor median/min/max/final robust-z residual versus prior same-time-of-day observations",
        "development_scenario_count": len(dev_ids),
        "development_event_count": len(usable_dev),
        "candidate_node_count": len(candidates),
        "raw_sensor_count": len(features or []),
        "signature_feature_count": int(x_train.shape[1]),
        "cross_validation": cv,
        "scikit_learn_version": sklearn_version,
        "label_boundary": "Holdout node IDs are not read by this prediction process.",
    }
    (public / "localization_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Localized {len(predictions)} sealed events across {len(hold_ids)} scenarios; candidates={len(candidates)}")


if __name__ == "__main__":
    main()
