from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import wntr

LTOWN_URL = "https://raw.githubusercontent.com/KIOS-Research/EPANET-Benchmarks/aaa7346607f080ecea7fae834944739b3385ba32/L-Town/L-TOWN.inp"
PRESSURE_SENSORS = [
    "n1","n4","n31","n54","n105","n114","n163","n188","n215","n229","n288",
    "n296","n332","n342","n410","n415","n429","n458","n469","n495","n506",
    "n516","n519","n549","n613","n636","n644","n679","n722","n726","n740",
    "n752","n769"
]
FLOW_SENSORS = ["PUMP_1", "p227", "p235"]
LEVEL_SENSORS = ["T1"]


def main() -> None:
    out = Path("ltown_probe_work")
    out.mkdir(exist_ok=True)
    inp = out / "L-TOWN.inp"
    urllib.request.urlretrieve(LTOWN_URL, inp)

    wn = wntr.network.WaterNetworkModel(str(inp))
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.time.duration = 48 * 3600
    wn.options.time.hydraulic_timestep = 15 * 60
    wn.options.time.report_timestep = 15 * 60
    wn.options.time.pattern_timestep = min(int(wn.options.time.pattern_timestep), 15 * 60)

    leak_pipe = "p257"
    leak_node_name = "p257_probe_leaknode"
    new_pipe_name = "p257_probe_B"
    wn = wntr.morph.split_pipe(wn, leak_pipe, new_pipe_name, leak_node_name)
    leak_node = wn.get_node(leak_node_name)
    diameter_m = 0.015
    area = math.pi * (diameter_m / 2.0) ** 2
    leak_node.add_leak(
        wn,
        area=area,
        discharge_coeff=0.75,
        start_time=24 * 3600,
        end_time=46 * 3600,
    )

    start = time.perf_counter()
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    elapsed = time.perf_counter() - start

    pressure = results.node["pressure"][PRESSURE_SENSORS]
    flow = results.link["flowrate"][FLOW_SENSORS]
    level = results.node["pressure"][LEVEL_SENSORS]
    sensor = np.concatenate(
        [
            pressure.to_numpy(dtype=np.float64),
            flow.to_numpy(dtype=np.float64) * 3600.0,
            level.to_numpy(dtype=np.float64),
        ],
        axis=1,
    )

    summary = {
        "network": "L-Town v1.2",
        "network_source": LTOWN_URL,
        "wntr_version": wntr.__version__,
        "junctions": wn.num_junctions,
        "pipes": wn.num_pipes,
        "pressure_sensors": len(PRESSURE_SENSORS),
        "flow_sensors": len(FLOW_SENSORS),
        "level_sensors": len(LEVEL_SENSORS),
        "sensor_columns": int(sensor.shape[1]),
        "timesteps": int(sensor.shape[0]),
        "hydraulic_timestep_minutes": 15,
        "duration_hours": 48,
        "leak_pipe": leak_pipe,
        "leak_diameter_m": diameter_m,
        "leak_start_hour": 24,
        "simulation_seconds": elapsed,
        "finite_sensor_fraction": float(np.isfinite(sensor).mean()),
        "pressure_min": float(np.nanmin(pressure.to_numpy())),
        "pressure_max": float(np.nanmax(pressure.to_numpy())),
    }
    (out / "probe.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
