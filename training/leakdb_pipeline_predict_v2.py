from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakdb_localize_predict import event_signature, rank_nodes
from leakdb_predict_v2 import causal_smooth
from leakdb_predict_v3 import fit_model, latch_alert, raw_probabilities
from leakdb_sealed import _load_bundle

LOCALIZATION_OBSERVATION_STEPS = 10
STRICT_WINDOW_STEPS = 10


def freeze_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(v) for v in payload["holdout_scenario_ids"]]


def challenge_ids(leakdb_dir: Path, number: int) -> list[int]:
    return sorted(freeze_ids(leakdb_dir / f"challenge_{number:03d}_prediction_freeze.json"))


def event_segments(y: np.ndarray) -> list[tuple[int, int]]:
    y = np.asarray(y, dtype=np.uint8)
    starts = np.flatnonzero((y == 1) & (np.r_[0, y[:-1]] == 0))
    ends = np.flatnonzero((y == 1) & (np.r_[y[1:], 0] == 0))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def alert_episode_starts(alert: np.ndarray) -> list[int]:
    alert = np.asarray(alert, dtype=np.uint8)
    return [int(v) for v in np.flatnonzero((alert == 1) & (np.r_[0, alert[:-1]] == 0))]


def alert_episode_end(alert: np.ndarray, start: int) -> int:
    i = start
    while i + 1 < len(alert) and alert[i + 1] == 1:
        i += 1
    return i


def make_alert(raw: np.ndarray, smooth: int, threshold: float, hold: int) -> np.ndarray:
    prob = causal_smooth(raw, smooth)
    trigger = (prob >= threshold).astype(np.uint8)
    return latch_alert(trigger, hold)


def policy_validation(
    ys: dict[int, np.ndarray],
    ids: list[int],
    raw_probs: dict[int, np.ndarray],
    smooth: int,
    threshold: float,
    hold: int,
) -> dict:
    event_hits = event_total = 0
    delays: list[int] = []
    no_leak_episodes = 0
    no_leak_days = 0.0
    unmatched_episodes = 0
    all_days = 0.0

    for sid in ids:
        y = np.asarray(ys[sid], dtype=np.uint8)
        alert = make_alert(raw_probs[sid], smooth, threshold, hold)
        starts = alert_episode_starts(alert)
        segments = event_segments(y)
        days = len(y) / 48.0
        all_days += days

        if not segments:
            no_leak_episodes += len(starts)
            no_leak_days += days

        for start in starts:
            if not any(seg_start <= start <= seg_end for seg_start, seg_end in segments):
                unmatched_episodes += 1

        for seg_start, seg_end in segments:
            event_total += 1
            deadline = min(seg_end, seg_start + STRICT_WINDOW_STEPS - 1)
            hits = [s for s in starts if seg_start <= s <= deadline]
            if hits:
                event_hits += 1
                delays.append(min(hits) - seg_start)

    early_recall = event_hits / event_total if event_total else 0.0
    no_leak_rate = no_leak_episodes / no_leak_days * 30.0 if no_leak_days else 0.0
    unmatched_rate = unmatched_episodes / all_days * 30.0 if all_days else 0.0
    return {
        "events": event_total,
        "early_new_alert_hits": event_hits,
        "early_new_alert_recall": early_recall,
        "median_start_delay_steps": float(np.median(delays)) if delays else None,
        "no_leak_alert_episodes_per_30d": no_leak_rate,
        "unmatched_alert_episodes_per_30d": unmatched_rate,
    }


def choose_policy(
    ys: dict[int, np.ndarray],
    groups: dict[str, list[int]],
    raw_probs: dict[int, np.ndarray],
) -> tuple[tuple[int, float, int], dict[str, dict]]:
    best = None
    for smooth in (1, 2, 3):
        for threshold in np.linspace(0.45, 0.90, 19):
            for hold in (1, 2, 4, 6, 8):
                results = {
                    name: policy_validation(ys, ids, raw_probs, smooth, float(threshold), hold)
                    for name, ids in groups.items()
                }
                early = [r["early_new_alert_recall"] for r in results.values()]
                no_leak = [r["no_leak_alert_episodes_per_30d"] for r in results.values()]
                unmatched = [r["unmatched_alert_episodes_per_30d"] for r in results.values()]
                feasible = max(no_leak) <= 1.0 and max(unmatched) <= 1.5
                fallback = min(
                    r["early_new_alert_recall"]
                    - 0.03 * r["no_leak_alert_episodes_per_30d"]
                    - 0.02 * r["unmatched_alert_episodes_per_30d"]
                    for r in results.values()
                )
                key = (
                    1 if feasible else 0,
                    min(early) if feasible else fallback,
                    sum(early) / len(early),
                    -max(no_leak),
                    -max(unmatched),
                    -hold,
                )
                candidate = (key, smooth, float(threshold), hold, results)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    assert best is not None
    return (best[1], best[2], best[3]), best[4]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibrated new-alert detector plus frozen localizer on pipeline v2 holdout."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline_v2_work"))
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

    groups: dict[str, list[int]] = {}
    detector_train_ids: set[int] = set(range(1, 21))
    for number in range(1, 6):
        ids = challenge_ids(leakdb_dir, number)
        groups[f"challenge_{number:03d}"] = ids[::2]
        detector_train_ids.update(ids[1::2])

    pipeline_001_ids = freeze_ids(leakdb_dir / "pipeline_001_prediction_freeze.json")
    groups["pipeline_001"] = sorted(pipeline_001_ids)

    missing = sorted(
        set(detector_train_ids).union(*(set(v) for v in groups.values())).difference(dev_ids)
    )
    if missing:
        raise RuntimeError(f"Detector development scenarios missing: {missing}")

    detector = fit_model(dev_x, dev_y, sorted(detector_train_ids))
    validation_ids = sorted(set().union(*(set(v) for v in groups.values())))
    val_raw = {sid: raw_probabilities(detector, dev_x[sid]) for sid in validation_ids}
    (smooth, threshold, hold), calibration = choose_policy(dev_y, groups, val_raw)

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
        "algorithm": "Pay Your Way Net1 pipeline v2",
        "detector": {
            "family": "HGB detector plus robust new-alert episode calibration",
            "training_scenario_ids": sorted(detector_train_ids),
            "validation_groups": groups,
            "alert_threshold": threshold,
            "alert_smooth_window": smooth,
            "alert_hold_steps": hold,
            "calibration_results": calibration,
            "objective": "maximize worst-cohort new-alert starts within ten half-hour steps subject to <=1 no-leak episode/30d and <=1.5 unmatched episodes/30d",
            "note": "Pipeline 001 is calibration-only and never detector training data.",
        },
        "localizer": {
            "family": "Frozen pipeline-001 ExtraTrees coordinate localizer v1",
            "training_event_count": len(localizer_events),
            "observation_steps_after_alert": LOCALIZATION_OBSERVATION_STEPS,
            "observation_hours_after_alert": LOCALIZATION_OBSERVATION_STEPS * 0.5,
            "candidate_node_count": len(candidates),
        },
        "holdout_scenario_count": len(hold_ids),
        "raw_sensor_count": len(features or []),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No pipeline-002 holdout leak timing or node labels are opened by prediction.",
    }
    (public / "pipeline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Pipeline v2 predicted {len(hold_ids)} sealed scenarios with {sum(len(r['alert_episodes']) for r in results)} episodes; "
        f"policy threshold={threshold:.3f}, smooth={smooth}, hold={hold}"
    )


if __name__ == "__main__":
    main()
