from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from leakdb_hanoi_acquire import DEV_IDS, TRAIN_IDS
from leakdb_localize_predict import event_signature, rank_nodes
from leakdb_predict_v2 import causal_smooth
from leakdb_predict_v3 import fit_model, raw_probabilities
from leakdb_sealed import _load_bundle

OPEN_THRESHOLD = 0.775
OPEN_SMOOTH = 2
LOCALIZATION_OBSERVATION_STEPS = 10
STRICT_WINDOW_STEPS = 10
STALE_INCIDENT_STEPS = 48  # 24 hours


def hysteresis_incidents(
    probability: np.ndarray,
    close_threshold: float,
    recovery_steps: int,
) -> list[tuple[int, int]]:
    probability = np.asarray(probability, dtype=np.float64)
    episodes: list[tuple[int, int]] = []
    active = False
    start = -1
    recovery = 0

    for i, value in enumerate(probability):
        if not active:
            if value >= OPEN_THRESHOLD:
                active = True
                start = i
                recovery = 0
            continue

        if value < close_threshold:
            recovery += 1
            if recovery >= recovery_steps:
                episodes.append((start, i))
                active = False
                start = -1
                recovery = 0
        else:
            recovery = 0

    if active:
        episodes.append((start, len(probability) - 1))
    return episodes


def overlaps_window(episode: tuple[int, int], start: int, end: int) -> bool:
    return not (episode[1] < start or episode[0] > end)


def calibration_metrics(
    scenario_ids: list[int],
    ys: dict[int, np.ndarray],
    raw: dict[int, np.ndarray],
    labels_by_sid: dict[int, list[dict]],
    close_threshold: float,
    recovery_steps: int,
) -> dict:
    total_events = fresh_hits = coverage_hits = stale_merges = 0
    overlap_counts: list[int] = []
    total_incidents = unmatched = no_leak_incidents = 0
    total_days = no_leak_days = 0.0

    for sid in scenario_ids:
        y = ys[sid]
        probability = causal_smooth(raw[sid], OPEN_SMOOTH)
        episodes = hysteresis_incidents(probability, close_threshold, recovery_steps)
        events = labels_by_sid.get(sid, [])
        total_incidents += len(episodes)
        days = len(y) / 48.0
        total_days += days
        if not events:
            no_leak_incidents += len(episodes)
            no_leak_days += days

        for episode in episodes:
            if not any(overlaps_window(episode, e["start_idx"], e["end_idx"]) for e in events):
                unmatched += 1

        for event in events:
            total_events += 1
            start = event["start_idx"]
            early_end = min(event["end_idx"], start + STRICT_WINDOW_STEPS - 1)
            fresh = [e for e in episodes if start <= e[0] <= early_end]
            covering = [e for e in episodes if overlaps_window(e, start, early_end)]
            all_overlaps = [e for e in episodes if overlaps_window(e, start, event["end_idx"])]
            overlap_counts.append(len(all_overlaps))
            if fresh:
                fresh_hits += 1
            if covering:
                coverage_hits += 1
            if not fresh:
                preexisting = [e for e in covering if e[0] < start]
                if preexisting and start - min(e[0] for e in preexisting) > STALE_INCIDENT_STEPS:
                    stale_merges += 1

    denom = total_events or 1
    return {
        "events": total_events,
        "fresh_start_recall": fresh_hits / denom,
        "early_coverage_recall": coverage_hits / denom,
        "stale_merge_fraction": stale_merges / denom,
        "mean_fragmentation": sum(overlap_counts) / len(overlap_counts) if overlap_counts else 0.0,
        "median_fragmentation": float(np.median(overlap_counts)) if overlap_counts else 0.0,
        "max_fragmentation": max(overlap_counts, default=0),
        "total_incidents": total_incidents,
        "unmatched_incidents_per_30d": unmatched / total_days * 30.0 if total_days else 0.0,
        "no_leak_incidents_per_30d": (
            no_leak_incidents / no_leak_days * 30.0 if no_leak_days else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Hanoi incident close hysteresis and predict sealed holdout.")
    parser.add_argument("--out-dir", type=Path, default=Path("hanoi_v3_work"))
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

    calibration_payload = json.loads(
        (out / "incident_calibration_labels.json").read_text(encoding="utf-8")
    )
    labels_by_group: dict[str, dict[int, list[dict]]] = {}
    all_calibration_ids: set[int] = set()
    for name, rows in calibration_payload.items():
        labels_by_group[name] = {int(row["scenario_id"]): row["events"] for row in rows}
        all_calibration_ids.update(labels_by_group[name])

    raw = {
        sid: raw_probabilities(detector, dev_x[sid])
        for sid in sorted(all_calibration_ids)
    }

    candidates = []
    for close_threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
        for recovery_steps in (12, 24, 48, 96, 192):
            groups = {}
            for name, labels_by_sid in labels_by_group.items():
                ids = sorted(labels_by_sid)
                groups[name] = calibration_metrics(
                    ids,
                    dev_y,
                    raw,
                    labels_by_sid,
                    close_threshold,
                    recovery_steps,
                )

            fresh = [g["fresh_start_recall"] for g in groups.values()]
            coverage = [g["early_coverage_recall"] for g in groups.values()]
            nuisance = [g["no_leak_incidents_per_30d"] for g in groups.values()]
            stale = [g["stale_merge_fraction"] for g in groups.values()]
            fragmentation = [g["mean_fragmentation"] for g in groups.values()]

            feasible = (
                min(fresh) >= 0.60
                and min(coverage) >= 0.65
                and max(nuisance) <= 0.50
                and max(stale) <= 0.10
            )
            key = (
                1 if feasible else 0,
                -max(fragmentation),
                min(fresh),
                min(coverage),
                -max(nuisance),
                -max(stale),
                close_threshold,
                recovery_steps,
            )
            candidates.append({
                "key": key,
                "close_threshold": close_threshold,
                "recovery_steps": recovery_steps,
                "feasible": feasible,
                "groups": groups,
            })

    best = max(candidates, key=lambda c: c["key"])
    close_threshold = float(best["close_threshold"])
    recovery_steps = int(best["recovery_steps"])

    topology = json.loads((public / "topology.json").read_text(encoding="utf-8"))
    coords = topology["coordinates"]
    candidate_nodes = topology["candidate_nodes"]
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
    for sid in hold_ids:
        x = hold_x[sid]
        probability = causal_smooth(raw_probabilities(detector, x), OPEN_SMOOTH)
        raw_episodes = hysteresis_incidents(probability, close_threshold, recovery_steps)
        episodes = []
        for start, end in raw_episodes:
            sig = event_signature(x, start, horizon=LOCALIZATION_OBSERVATION_STEPS)
            pred_xy = localizer.predict(sig.reshape(1, -1))[0]
            ranking = rank_nodes(pred_xy, coords, candidate_nodes)
            episodes.append({
                "alert_start_idx": start,
                "alert_end_idx": end,
                "localization_ready_idx": min(
                    len(x) - 1, start + LOCALIZATION_OBSERVATION_STEPS - 1
                ),
                "predicted_xy": [float(pred_xy[0]), float(pred_xy[1])],
                "ranked_nodes": ranking,
            })
        results.append({
            "scenario_id": sid,
            "timesteps": int(len(x)),
            "alert_episodes": episodes,
        })

    (public / "pipeline_predictions.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "algorithm": "Pay Your Way LeakDB Hanoi pipeline v3 calibrated incident hysteresis",
        "detector": {
            "family": "identical Hanoi pipeline-001 HGB detector",
            "training_scenario_ids": TRAIN_IDS,
            "open_threshold": OPEN_THRESHOLD,
            "open_smooth_window": OPEN_SMOOTH,
        },
        "incident_state": {
            "close_threshold": close_threshold,
            "recovery_steps": recovery_steps,
            "recovery_hours": recovery_steps * 0.5,
            "selection_rule": "minimize worst burned-cohort fragmentation subject to fresh-start, coverage, nuisance, and stale-merge constraints",
            "calibration_groups": best["groups"],
            "feasible": best["feasible"],
        },
        "localizer": {
            "family": "identical Hanoi pipeline-001 ExtraTrees coordinate localizer",
            "training_scenario_ids": DEV_IDS,
            "training_event_count": len(localizer_events),
            "candidate_node_count": len(candidate_nodes),
            "observation_steps_after_alert": LOCALIZATION_OBSERVATION_STEPS,
        },
        "holdout_scenario_count": len(hold_ids),
        "raw_sensor_count": len(features or []),
        "scikit_learn_version": sklearn_version,
        "label_boundary": "No Hanoi-003 holdout leak timing or node labels are opened by prediction.",
    }
    (public / "pipeline_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Hanoi v3 predicted {len(hold_ids)} sealed scenarios; "
        f"close={close_threshold:.2f}, recovery={recovery_steps} steps; "
        f"incidents={sum(len(r['alert_episodes']) for r in results)}"
    )


if __name__ == "__main__":
    main()
