from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import numpy as np

from leakdb_acquire_v2 import load_frames
from leakdb_acquire_v3 import burned_ids
from leakdb_sealed import _save_bundle


def parse_inp(path: Path) -> dict:
    section = ""
    coords: dict[str, list[float]] = {}
    junctions: list[str] = []
    edges: list[dict] = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        parts = line.split()
        if section == "JUNCTIONS" and len(parts) >= 1:
            junctions.append(parts[0])
        elif section == "COORDINATES" and len(parts) >= 3:
            try:
                coords[parts[0]] = [float(parts[1]), float(parts[2])]
            except ValueError:
                pass
        elif section == "PIPES" and len(parts) >= 4:
            try:
                length = float(parts[3])
            except ValueError:
                length = 1.0
            edges.append({"id": parts[0], "node1": parts[1], "node2": parts[2], "length": length, "type": "pipe"})
        elif section == "PUMPS" and len(parts) >= 3:
            edges.append({"id": parts[0], "node1": parts[1], "node2": parts[2], "length": 1.0, "type": "pump"})
        elif section == "VALVES" and len(parts) >= 3:
            edges.append({"id": parts[0], "node1": parts[1], "node2": parts[2], "length": 1.0, "type": "valve"})

    candidates = [node for node in junctions if node in coords]
    if not candidates:
        raise RuntimeError("Net1 topology did not expose junction coordinates")
    return {"coordinates": coords, "candidate_nodes": candidates, "edges": edges}


def leak_catalog() -> dict[str, list[dict]]:
    from water_benchmark_hub.leakdb.leakdb_data import NET1_LEAKAGES
    raw = json.loads(NET1_LEAKAGES)
    catalog: dict[str, list[dict]] = {}
    for sid, events in raw.items():
        cleaned = []
        for index, event in enumerate(events):
            cleaned.append({
                "event_index": index,
                "node_id": str(event["node_id"]),
                "start_idx": int(event["leak_start_time"]),
                "end_idx": int(event["leak_end_time"]),
                "leak_diameter": float(event.get("leak_diameter", 0.0)),
            })
        catalog[str(sid)] = cleaned
    return catalog


def challenge_source_by_scenario(leakdb_dir: Path, burned_through: int) -> dict[int, str]:
    sources = {sid: "base" for sid in range(1, 21)}
    for number in range(1, burned_through + 1):
        payload = json.loads(
            (leakdb_dir / f"challenge_{number:03d}_prediction_freeze.json").read_text(encoding="utf-8")
        )
        for sid in payload["holdout_scenario_ids"]:
            sources[int(sid)] = f"challenge_{number:03d}"
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sealed conditional LeakDB localization benchmark.")
    parser.add_argument("--out-dir", type=Path, default=Path("localization_work"))
    parser.add_argument("--holdout-count", type=int, default=50)
    parser.add_argument("--holdout-min", type=int, default=101)
    parser.add_argument("--holdout-max", type=int, default=1000)
    parser.add_argument("--burned-through", type=int, default=5)
    parser.add_argument("--challenge-id", default="leakdb-net1-localize-001")
    args = parser.parse_args()

    out = args.out_dir
    public, secret, cache = out / "public", out / "secret", out / "cache"
    public.mkdir(parents=True, exist_ok=True)
    secret.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    leakdb_dir = Path(__file__).resolve().parent / "leakdb"
    burned, _ = burned_ids(leakdb_dir, args.burned_through)
    sources = challenge_source_by_scenario(leakdb_dir, args.burned_through)
    catalog = leak_catalog()

    from water_benchmark_hub import load
    network = load("Network-Net1")
    inp_path = Path(network.load(download_dir=str(cache), verbose=False, return_scenario=False))
    topology = parse_inp(inp_path)
    candidate_set = set(topology["candidate_nodes"])

    def usable_events(sid: int) -> list[dict]:
        return [e for e in catalog.get(str(sid), []) if e["node_id"] in candidate_set]

    dev_ids = sorted(sid for sid in burned if usable_events(sid))
    burned_set = set(burned)
    candidate_scenarios = [
        sid for sid in range(args.holdout_min, args.holdout_max + 1)
        if sid not in burned_set and usable_events(sid)
    ]
    if len(candidate_scenarios) < args.holdout_count:
        raise RuntimeError("Not enough unburned leaking scenarios for localization holdout")
    holdout_ids = sorted(secrets.SystemRandom().sample(candidate_scenarios, args.holdout_count))

    print(f"Localization: {len(dev_ids)} burned leaking development scenarios; {len(holdout_ids)} fresh sealed scenarios.")
    dev_x, _, features = load_frames(dev_ids, cache)
    hold_x, _, hold_features = load_frames(holdout_ids, cache)
    if features != hold_features:
        raise RuntimeError("Development and holdout feature schemas differ")

    _save_bundle(out / "dev_inputs.npz", dev_ids, dev_x, None, features)
    _save_bundle(out / "holdout_inputs.npz", holdout_ids, hold_x, None, features)

    dev_events: list[dict] = []
    for sid in dev_ids:
        for event in usable_events(sid):
            dev_events.append({
                "event_id": f"{sid}:{event['event_index']}",
                "scenario_id": sid,
                "source_group": sources.get(sid, "burned"),
                **event,
            })

    public_events: list[dict] = []
    secret_events: list[dict] = []
    for sid in holdout_ids:
        for event in usable_events(sid):
            event_id = f"{sid}:{event['event_index']}"
            public_events.append({
                "event_id": event_id,
                "scenario_id": sid,
                "event_index": event["event_index"],
                "start_idx": event["start_idx"],
                "end_idx": event["end_idx"],
            })
            secret_events.append({
                "event_id": event_id,
                "scenario_id": sid,
                "event_index": event["event_index"],
                "node_id": event["node_id"],
                "leak_diameter": event["leak_diameter"],
            })

    (out / "dev_localization_labels.json").write_text(json.dumps(dev_events, indent=2) + "\n", encoding="utf-8")
    (out / "holdout_events.json").write_text(json.dumps(public_events, indent=2) + "\n", encoding="utf-8")
    (public / "topology.json").write_text(json.dumps(topology, indent=2) + "\n", encoding="utf-8")
    (secret / "location_labels.json").write_text(json.dumps(secret_events, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "challenge_id": args.challenge_id,
        "task": "conditional localization given leak-event onset; leak node withheld",
        "benchmark": "KIOS-LeakDB",
        "network": "Net1",
        "burned_through_detection_challenge": args.burned_through,
        "development_scenario_count": len(dev_ids),
        "development_event_count": len(dev_events),
        "holdout_scenario_ids": holdout_ids,
        "holdout_event_count": len(public_events),
        "holdout_selection": "cryptographically random sample of unburned scenarios with at least one coordinate-resolvable leak event",
        "candidate_node_count": len(topology["candidate_nodes"]),
        "feature_count": len(features),
        "public_information": "sensor time series, event onset/end indices, and network topology/coordinates",
        "withheld_information": "leak node IDs and leak diameters for holdout events",
    }
    (public / "localization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
