from __future__ import annotations

import argparse
import json
import random
import secrets
from pathlib import Path

from ltown_sealed import (
    PRESSURE_SENSORS,
    download_network,
    network_zone_map,
    save_bundle,
    simulate,
)

DEV_SEED = 20260918


def leak_spec(
    scenario_id: int,
    zone: str,
    pipes: list[str],
    rng: random.Random,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "kind": "leak",
        "pipe_id": rng.choice(pipes),
        "zone_id": zone,
        "diameter_m": rng.uniform(0.007, 0.027),
        "start_hour": rng.uniform(26.0, 34.0),
        "seed": rng.randrange(0, 2**32),
    }


def normal_spec(scenario_id: int, rng: random.Random) -> dict:
    return {
        "scenario_id": scenario_id,
        "kind": "normal",
        "seed": rng.randrange(0, 2**32),
    }


def build_specs(
    by_zone: dict[str, list[str]],
    holdout_count: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    dev_rng = random.Random(DEV_SEED)
    zones = [zone for zone in PRESSURE_SENSORS if by_zone.get(zone)]
    detector_train: list[dict] = []
    localizer_dev: list[dict] = []
    sid = 1

    # Detection training remains moderate; localization development is disjoint
    # so its alert-time signatures are generated out-of-sample from the detector.
    for zone in zones:
        for _ in range(2):
            detector_train.append(
                leak_spec(sid, zone, by_zone[zone], dev_rng)
            )
            sid += 1

    for _ in range(10):
        detector_train.append(normal_spec(sid, dev_rng))
        sid += 1

    for zone in zones:
        for _ in range(3):
            localizer_dev.append(
                leak_spec(sid, zone, by_zone[zone], dev_rng)
            )
            sid += 1

    for _ in range(10):
        localizer_dev.append(normal_spec(sid, dev_rng))
        sid += 1

    holdout: list[dict] = []
    crypto = secrets.SystemRandom()
    for _ in range(holdout_count):
        if crypto.random() < 0.20:
            holdout.append(
                {
                    "scenario_id": sid,
                    "kind": "normal",
                    "seed": secrets.randbits(32),
                }
            )
        else:
            zone = crypto.choice(zones)
            holdout.append(
                {
                    "scenario_id": sid,
                    "kind": "leak",
                    "pipe_id": crypto.choice(by_zone[zone]),
                    "zone_id": zone,
                    "diameter_m": crypto.uniform(0.006, 0.028),
                    "start_hour": crypto.uniform(26.0, 34.0),
                    "seed": secrets.randbits(32),
                }
            )
        sid += 1

    return detector_train, localizer_dev, holdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible L-Town development data plus sealed holdout."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("ltown_v2_work"))
    parser.add_argument("--holdout-count", type=int, default=50)
    parser.add_argument("--challenge-id", default="ltown-sealed-002")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)

    inp = download_network(cache)
    pipe_to_zone, by_zone = network_zone_map(inp)
    detector_train, localizer_dev, holdout = build_specs(
        by_zone, args.holdout_count
    )

    all_specs = detector_train + localizer_dev + holdout
    xs = {}
    ys = {}
    for index, spec in enumerate(all_specs, 1):
        sid = spec["scenario_id"]
        print(
            f"simulate {index}/{len(all_specs)} scenario={sid} kind={spec['kind']}",
            flush=True,
        )
        x, y = simulate(inp, spec, int(spec["seed"]))
        xs[sid] = x
        ys[sid] = y

    save_bundle(out / "detector_train_bundle.npz", detector_train, xs, ys)
    save_bundle(out / "localizer_dev_bundle.npz", localizer_dev, xs, ys)
    save_bundle(out / "holdout_inputs.npz", holdout, xs, None)

    (out / "detector_train_specs.json").write_text(
        json.dumps(detector_train, indent=2) + "\n", encoding="utf-8"
    )
    (out / "localizer_dev_specs.json").write_text(
        json.dumps(localizer_dev, indent=2) + "\n", encoding="utf-8"
    )
    (secret / "holdout_labels.json").write_text(
        json.dumps(
            [{k: v for k, v in spec.items() if k != "seed"} for spec in holdout],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (public / "holdout_manifest.json").write_text(
        json.dumps([{"scenario_id": spec["scenario_id"]} for spec in holdout], indent=2)
        + "\n",
        encoding="utf-8",
    )
    (public / "zone_map.json").write_text(
        json.dumps(
            {"pressure_sensors": PRESSURE_SENSORS, "pipe_to_zone": pipe_to_zone},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (public / "challenge.json").write_text(
        json.dumps(
            {
                "challenge_id": args.challenge_id,
                "network": "L-Town v1.2",
                "junctions": 783,
                "pipes": 906,
                "sensor_columns": 37,
                "pressure_search_zones": len(
                    [z for z in PRESSURE_SENSORS if by_zone.get(z)]
                ),
                "development_seed": DEV_SEED,
                "detector_train_scenarios": len(detector_train),
                "localizer_dev_scenarios": len(localizer_dev),
                "holdout_scenarios": len(holdout),
                "development_design":
                    "2 leak scenarios per zone for detector training; "
                    "3 disjoint leak scenarios per zone for alert-aligned localizer development",
                "timestep_minutes": 15,
                "scenario_hours": 48,
                "label_boundary":
                    "holdout leak/no-leak, pipe, zone, diameter, and start time "
                    "stored only in the sealed label artifact",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
