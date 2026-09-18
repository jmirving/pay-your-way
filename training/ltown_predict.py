from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAL_STEPS = 96
PERIOD = 96
LOCALIZE_STEPS = 20
DETECTION_DEADLINE_STEPS = 20


def load_bundle(path: Path, labels: bool):
    z = np.load(path, allow_pickle=False)
    ids = [int(v) for v in z["scenario_ids"]]
    xs = {sid: z[f"x_{sid}"].astype(np.float32) for sid in ids}
    ys = (
        {sid: z[f"y_{sid}"].astype(np.uint8) for sid in ids}
        if labels
        else None
    )
    return ids, xs, ys


def features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    cal = x[:CAL_STEPS]
    center = np.nanmedian(cal, axis=0)
    mad = np.nanmedian(np.abs(cal - center), axis=0)
    scale = 1.4826 * mad
    std = np.nanstd(cal, axis=0)
    scale = np.where(
        scale > 1e-6,
        scale,
        np.where(std > 1e-6, std, 1.0),
    )

    local = np.clip(
        np.nan_to_num(
            (x - center) / scale,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ),
        -20,
        20,
    )

    expected = np.empty_like(x)
    expected[:PERIOD] = np.tile(
        np.nanmedian(cal, axis=0),
        (PERIOD, 1),
    )
    expected[PERIOD:] = x[:-PERIOD]
    seasonal = np.clip(
        np.nan_to_num(
            (x - expected) / scale,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ),
        -20,
        20,
    )

    d1 = np.zeros_like(seasonal)
    d1[1:] = seasonal[1:] - seasonal[:-1]

    return np.concatenate(
        [local, seasonal, d1],
        axis=1,
    ).astype(np.float32)


def smooth(p: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return p
    c = np.cumsum(np.r_[0.0, p])
    n = np.arange(1, len(p) + 1)
    starts = np.maximum(0, n - window)
    return (c[n] - c[starts]) / (n - starts)


def onset(y: np.ndarray) -> int | None:
    idx = np.flatnonzero(
        (y == 1) & (np.r_[0, y[:-1]] == 0)
    )
    return int(idx[0]) if len(idx) else None


def first_alert(
    prob: np.ndarray,
    threshold: float,
) -> int | None:
    idx = np.flatnonzero(
        (np.arange(len(prob)) >= CAL_STEPS)
        & (prob >= threshold)
    )
    return int(idx[0]) if len(idx) else None


def loc_signature(
    x: np.ndarray,
    start: int,
) -> np.ndarray:
    end = min(len(x), start + LOCALIZE_STEPS)
    current = x[start:end].astype(np.float64)
    if not len(current):
        return np.zeros(
            x.shape[1] * 4,
            dtype=np.float32,
        )

    previous_start = start - PERIOD
    if (
        previous_start >= 0
        and previous_start + len(current) <= len(x)
    ):
        expected = x[
            previous_start : previous_start + len(current)
        ].astype(np.float64)
    else:
        expected = np.repeat(
            np.nanmedian(
                x[:CAL_STEPS],
                axis=0,
                keepdims=True,
            ),
            len(current),
            axis=0,
        )

    pre = x[: max(start, 1)].astype(np.float64)
    center = np.nanmedian(pre, axis=0)
    mad = np.nanmedian(
        np.abs(pre - center),
        axis=0,
    )
    scale = 1.4826 * mad
    std = np.nanstd(pre, axis=0)
    scale = np.where(
        scale > 1e-6,
        scale,
        np.where(std > 1e-6, std, 1.0),
    )

    residual = np.clip(
        np.nan_to_num(
            (current - expected) / scale,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ),
        -20,
        20,
    )

    return np.concatenate(
        [
            np.nanmedian(residual, axis=0),
            np.nanmin(residual, axis=0),
            np.nanmax(residual, axis=0),
            residual[-1],
        ]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ltown_work"),
    )
    args = parser.parse_args()

    out = args.out_dir
    public = out / "public"

    train_ids, train_x, train_y = load_bundle(
        out / "train_bundle.npz",
        True,
    )
    val_ids, val_x, val_y = load_bundle(
        out / "validation_bundle.npz",
        True,
    )
    hold_ids, hold_x, _ = load_bundle(
        out / "holdout_inputs.npz",
        False,
    )

    train_specs = {
        row["scenario_id"]: row
        for row in json.loads(
            (out / "train_specs.json").read_text()
        )
    }
    val_specs = {
        row["scenario_id"]: row
        for row in json.loads(
            (out / "validation_specs.json").read_text()
        )
    }
    all_specs = train_specs | val_specs

    from sklearn import __version__ as sklearn_version
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
    )

    rng = np.random.default_rng(0)
    chunks_x = []
    chunks_y = []
    for sid in train_ids:
        f = features(train_x[sid])
        y = train_y[sid]
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(y == 0)
        keep_neg = rng.choice(
            neg,
            size=min(
                len(neg),
                max(50, len(pos) * 4),
            ),
            replace=False,
        )
        idx = np.sort(
            np.r_[pos, keep_neg]
        )
        chunks_x.append(f[idx])
        chunks_y.append(y[idx])

    detector = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.06,
        max_depth=7,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=0,
    )
    detector.fit(
        np.concatenate(chunks_x),
        np.concatenate(chunks_y),
    )

    raw = {
        sid: detector.predict_proba(
            features(val_x[sid])
        )[:, 1]
        for sid in val_ids
    }

    best = None
    for window in (1, 2, 4, 8):
        for threshold in np.linspace(
            0.40,
            0.97,
            24,
        ):
            hits = leaks = premature = 0
            false_controls = controls = 0

            for sid in val_ids:
                alert = first_alert(
                    smooth(raw[sid], window),
                    float(threshold),
                )
                event_onset = onset(val_y[sid])

                if event_onset is None:
                    controls += 1
                    false_controls += alert is not None
                else:
                    leaks += 1
                    if (
                        alert is not None
                        and alert < event_onset
                    ):
                        premature += 1
                    if (
                        alert is not None
                        and event_onset
                        <= alert
                        <= event_onset
                        + DETECTION_DEADLINE_STEPS
                    ):
                        hits += 1

            false_fraction = (
                false_controls / max(controls, 1)
            )
            premature_fraction = (
                premature / max(leaks, 1)
            )
            recall = hits / max(leaks, 1)
            feasible = (
                false_fraction <= 0.15
                and premature_fraction <= 0.10
            )
            key = (
                1 if feasible else 0,
                recall,
                -false_fraction,
                -premature_fraction,
                float(threshold),
                -window,
            )
            candidate = (
                key,
                window,
                float(threshold),
                {
                    "early_recall": recall,
                    "no_leak_false_scenario_fraction":
                        false_fraction,
                    "premature_alert_fraction":
                        premature_fraction,
                    "leaks": leaks,
                    "controls": controls,
                },
            )
            if (
                best is None
                or candidate[0] > best[0]
            ):
                best = candidate

    assert best is not None
    _, smooth_window, threshold, validation = best

    localizer_x = []
    localizer_y = []
    dev_y = train_y | val_y
    for sid, x in (
        list(train_x.items())
        + list(val_x.items())
    ):
        spec = all_specs[sid]
        if spec["kind"] != "leak":
            continue
        event_onset = onset(dev_y[sid])
        localizer_x.append(
            loc_signature(
                x,
                event_onset,
            )
        )
        localizer_y.append(
            spec["zone_id"]
        )

    localizer = ExtraTreesClassifier(
        n_estimators=1000,
        min_samples_leaf=1,
        max_features=0.75,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )
    localizer.fit(
        np.stack(localizer_x),
        np.asarray(localizer_y),
    )

    results = []
    for sid in hold_ids:
        probability = smooth(
            detector.predict_proba(
                features(hold_x[sid])
            )[:, 1],
            smooth_window,
        )
        alert = first_alert(
            probability,
            threshold,
        )

        row = {
            "scenario_id": sid,
            "timesteps": int(
                len(probability)
            ),
            "alert_start_idx": alert,
            "localization_ready_idx": None,
            "top_zones": [],
        }

        if alert is not None:
            ready = min(
                len(probability) - 1,
                alert
                + LOCALIZE_STEPS
                - 1,
            )
            signature = loc_signature(
                hold_x[sid],
                alert,
            )
            probabilities = (
                localizer.predict_proba(
                    signature.reshape(1, -1)
                )[0]
            )
            order = np.argsort(
                probabilities
            )[::-1][:5]

            row[
                "localization_ready_idx"
            ] = ready
            row["top_zones"] = [
                {
                    "zone_id": str(
                        localizer.classes_[i]
                    ),
                    "probability": float(
                        probabilities[i]
                    ),
                }
                for i in order
            ]

        results.append(row)

    (public / "predictions.json").write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n"
    )

    (public / "prediction_report.json").write_text(
        json.dumps(
            {
                "algorithm":
                    "Pay Your Way L-Town sparse-sensor baseline v1",
                "detector": {
                    "smooth_window":
                        smooth_window,
                    "threshold":
                        threshold,
                    "validation":
                        validation,
                    "training_scenarios":
                        train_ids,
                    "validation_scenarios":
                        val_ids,
                },
                "localizer": {
                    "target":
                        "nearest pressure-sensor graph zone",
                    "training_leak_scenarios":
                        len(localizer_y),
                    "zone_classes":
                        len(localizer.classes_),
                    "top_k": 5,
                    "observation_hours_after_alert":
                        5.0,
                },
                "holdout_scenarios":
                    len(hold_ids),
                "raw_sensor_count": 37,
                "scikit_learn_version":
                    sklearn_version,
                "label_boundary":
                    "No holdout labels are opened by prediction.",
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"predicted {len(hold_ids)} holdout; "
        f"threshold={threshold:.3f} "
        f"smooth={smooth_window}; "
        f"zones={len(localizer.classes_)}"
    )


if __name__ == "__main__":
    main()
