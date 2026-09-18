from __future__ import annotations

import argparse
import json
import math
import secrets
import urllib.request
from pathlib import Path

import numpy as np

LTOWN_URL = "https://raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/aaa7346607f080ecea7fae834944739b3385ba32/L-Town/L-TOWN.inp"
PRESSURE_SENSORS = [
    "n1","n4","n31","n54","n105","n114","n163","n188","n215","n229","n288",
    "n296","n332","n342","n410","n415","n429","n458","n469","n495","n506",
    "n516","n519","n549","n613","n636","n644","n679","n722","n726","n740",
    "n752","n769"
]
FLOW_SENSORS = ["PUMP_1", "p227", "p235"]
LEVEL_SENSORS = ["T1"]
STEP_SECONDS = 15 * 60
PERIOD_STEPS = 96
TIMESTEPS = 193
LEAK_END_HOUR = 46.0
EXCLUDED_LEAK_PIPES = {"p227", "p235"}


def download_network(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "L-TOWN.inp"
    if not path.exists():
        urllib.request.urlretrieve(LTOWN_URL, path)
    return path


def network_zone_map(inp: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    import networkx as nx
    import wntr

    wn = wntr.network.WaterNetworkModel(str(inp))
    graph = nx.Graph()
    for name, link in wn.links():
        u, v = link.start_node_name, link.end_node_name
        length = float(getattr(link, "length", 1.0) or 1.0)
        graph.add_edge(u, v, weight=max(length, 1.0))
    distances = {
        sensor: nx.single_source_dijkstra_path_length(graph, sensor, weight="weight")
        for sensor in PRESSURE_SENSORS
    }
    pipe_to_zone: dict[str, str] = {}
    by_zone: dict[str, list[str]] = {sensor: [] for sensor in PRESSURE_SENSORS}
    for name, pipe in wn.pipes():
        if name in EXCLUDED_LEAK_PIPES:
            continue
        u, v = pipe.start_node_name, pipe.end_node_name
        half = float(pipe.length) / 2.0
        zone = min(
            PRESSURE_SENSORS,
            key=lambda s: min(
                distances[s].get(u, float("inf")),
                distances[s].get(v, float("inf")),
            ) + half,
        )
        pipe_to_zone[name] = zone
        by_zone[zone].append(name)
    return pipe_to_zone, by_zone


def configure_network(wn) -> None:
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.time.duration = 48 * 3600
    wn.options.time.hydraulic_timestep = STEP_SECONDS
    wn.options.time.report_timestep = STEP_SECONDS
    wn.options.time.pattern_timestep = min(
        int(wn.options.time.pattern_timestep), STEP_SECONDS
    )


def simulate(inp: Path, spec: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import wntr

    rng = np.random.default_rng(seed)
    wn = wntr.network.WaterNetworkModel(str(inp))
    configure_network(wn)

    leak_start_idx = None
    leak_end_idx = None
    if spec["kind"] == "leak":
        pipe_id = spec["pipe_id"]
        node_name = f"{pipe_id}_scenario_leaknode"
        new_pipe = f"{pipe_id}_scenario_B"
        wn = wntr.morph.split_pipe(wn, pipe_id, new_pipe, node_name)
        node = wn.get_node(node_name)
        diameter = float(spec["diameter_m"])
        area = math.pi * (diameter / 2.0) ** 2
        start_seconds = int(float(spec["start_hour"]) * 3600)
        end_seconds = int(LEAK_END_HOUR * 3600)
        node.add_leak(
            wn,
            area=area,
            discharge_coeff=0.75,
            start_time=start_seconds,
            end_time=end_seconds,
        )
        leak_start_idx = int(round(start_seconds / STEP_SECONDS))
        leak_end_idx = int(round(end_seconds / STEP_SECONDS))

    results = wntr.sim.WNTRSimulator(wn).run_sim()
    pressure = results.node["pressure"][PRESSURE_SENSORS].to_numpy(dtype=np.float64)
    flow = (
        results.link["flowrate"][FLOW_SENSORS].to_numpy(dtype=np.float64) * 3600.0
    )
    level = results.node["pressure"][LEVEL_SENSORS].to_numpy(dtype=np.float64)
    x = np.concatenate([pressure, flow, level], axis=1)

    # Scenario-specific instrument offset/noise makes local calibration necessary.
    x[:, : len(PRESSURE_SENSORS)] += rng.normal(
        0.0, 0.12, size=(1, len(PRESSURE_SENSORS))
    )
    x[:, : len(PRESSURE_SENSORS)] += rng.normal(
        0.0, 0.03, size=(len(x), len(PRESSURE_SENSORS))
    )
    flow_start = len(PRESSURE_SENSORS)
    flow_end = flow_start + len(FLOW_SENSORS)
    flow_scale = np.maximum(
        np.nanstd(x[:, flow_start:flow_end], axis=0), 0.1
    )
    x[:, flow_start:flow_end] += rng.normal(
        0.0,
        flow_scale * 0.01,
        size=(len(x), len(FLOW_SENSORS)),
    )
    x[:, -1:] += rng.normal(0.0, 0.02, size=(len(x), 1))

    y = np.zeros(len(x), dtype=np.uint8)
    if leak_start_idx is not None:
        y[leak_start_idx : min(len(y), leak_end_idx + 1)] = 1
    return x.astype(np.float32), y


def build_specs(
    by_zone: dict[str, list[str]],
    holdout_count: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = secrets.SystemRandom()
    zones = [z for z in PRESSURE_SENSORS if by_zone.get(z)]
    train, val = [], []
    sid = 1

    for zone in zones:
        for target in (train, val):
            pipe = rng.choice(by_zone[zone])
            target.append(
                {
                    "scenario_id": sid,
                    "kind": "leak",
                    "pipe_id": pipe,
                    "zone_id": zone,
                    "diameter_m": rng.uniform(0.008, 0.025),
                    "start_hour": rng.uniform(26.0, 34.0),
                    "seed": secrets.randbits(32),
                }
            )
            sid += 1

    for target in (train, val):
        for _ in range(7):
            target.append(
                {
                    "scenario_id": sid,
                    "kind": "normal",
                    "seed": secrets.randbits(32),
                }
            )
            sid += 1

    holdout = []
    for _ in range(holdout_count):
        if rng.random() < 0.20:
            holdout.append(
                {
                    "scenario_id": sid,
                    "kind": "normal",
                    "seed": secrets.randbits(32),
                }
            )
        else:
            zone = rng.choice(zones)
            pipe = rng.choice(by_zone[zone])
            holdout.append(
                {
                    "scenario_id": sid,
                    "kind": "leak",
                    "pipe_id": pipe,
                    "zone_id": zone,
                    "diameter_m": rng.uniform(0.006, 0.027),
                    "start_hour": rng.uniform(26.0, 34.0),
                    "seed": secrets.randbits(32),
                }
            )
        sid += 1

    return train, val, holdout


def save_bundle(
    path: Path,
    specs: list[dict],
    xs: dict[int, np.ndarray],
    ys: dict[int, np.ndarray] | None,
) -> None:
    payload: dict[str, object] = {
        "scenario_ids": np.asarray(
            [s["scenario_id"] for s in specs], dtype=np.int32
        )
    }
    for spec in specs:
        sid = spec["scenario_id"]
        payload[f"x_{sid}"] = xs[sid]
        if ys is not None:
            payload[f"y_{sid}"] = ys[sid]
    np.savez_compressed(path, **payload)


def prepare(args) -> None:
    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)

    inp = download_network(cache)
    pipe_to_zone, by_zone = network_zone_map(inp)
    train, val, holdout = build_specs(by_zone, args.holdout_count)

    all_specs = train + val + holdout
    xs: dict[int, np.ndarray] = {}
    ys: dict[int, np.ndarray] = {}

    for i, spec in enumerate(all_specs, 1):
        sid = spec["scenario_id"]
        print(
            f"simulate {i}/{len(all_specs)} "
            f"scenario={sid} kind={spec['kind']}",
            flush=True,
        )
        x, y = simulate(inp, spec, int(spec["seed"]))
        xs[sid] = x
        ys[sid] = y

    save_bundle(out / "train_bundle.npz", train, xs, ys)
    save_bundle(out / "validation_bundle.npz", val, xs, ys)
    save_bundle(out / "holdout_inputs.npz", holdout, xs, None)

    public_specs = []
    secret_labels = []
    for spec in holdout:
        public_specs.append({"scenario_id": spec["scenario_id"]})
        secret_labels.append(
            {k: v for k, v in spec.items() if k != "seed"}
        )

    (public / "holdout_manifest.json").write_text(
        json.dumps(public_specs, indent=2) + "\n"
    )
    (secret / "holdout_labels.json").write_text(
        json.dumps(secret_labels, indent=2) + "\n"
    )
    (out / "train_specs.json").write_text(
        json.dumps(train, indent=2) + "\n"
    )
    (out / "validation_specs.json").write_text(
        json.dumps(val, indent=2) + "\n"
    )
    (public / "zone_map.json").write_text(
        json.dumps(
            {
                "pressure_sensors": PRESSURE_SENSORS,
                "pipe_to_zone": pipe_to_zone,
            },
            indent=2,
        )
        + "\n"
    )
    (public / "challenge.json").write_text(
        json.dumps(
            {
                "challenge_id": args.challenge_id,
                "network": "L-Town v1.2",
                "network_source": LTOWN_URL,
                "junctions": 783,
                "pipes": 906,
                "sensor_columns": 37,
                "pressure_search_zones": len(
                    [z for z in PRESSURE_SENSORS if by_zone.get(z)]
                ),
                "training_scenarios": len(train),
                "validation_scenarios": len(val),
                "holdout_scenarios": len(holdout),
                "timestep_minutes": 15,
                "scenario_hours": 48,
                "leak_start_window_hours": [26, 34],
                "leak_end_hour": LEAK_END_HOUR,
                "label_boundary":
                    "holdout leak/no-leak status, pipe, zone, magnitude, "
                    "start time stored only in sealed labels artifact",
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ltown_work"),
    )
    parser.add_argument("--holdout-count", type=int, default=30)
    parser.add_argument(
        "--challenge-id",
        default="ltown-sealed-001",
    )
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
