from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path


def graph(topology: dict) -> dict[str, list[tuple[str, float]]]:
    result: dict[str, list[tuple[str, float]]] = {}
    for edge in topology["edges"]:
        a, b = edge["node1"], edge["node2"]
        w = max(float(edge.get("length", 1.0)), 1e-6)
        result.setdefault(a, []).append((b, w))
        result.setdefault(b, []).append((a, w))
    return result


def shortest(g: dict[str, list[tuple[str, float]]], start: str, goal: str, weighted: bool) -> float | None:
    if start == goal:
        return 0.0
    queue = [(0.0, start)]
    seen: dict[str, float] = {start: 0.0}
    while queue:
        dist, node = heapq.heappop(queue)
        if dist != seen.get(node):
            continue
        if node == goal:
            return dist
        for nxt, weight in g.get(node, []):
            nd = dist + (weight if weighted else 1.0)
            if nd < seen.get(nxt, float("inf")):
                seen[nxt] = nd
                heapq.heappush(queue, (nd, nxt))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frozen conditional LeakDB localization rankings after node-label reveal.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("localization_score.json"))
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    by_event = {row["event_id"]: row for row in labels}
    if {row["event_id"] for row in predictions} != set(by_event):
        raise RuntimeError("Prediction and label event IDs differ")

    g = graph(topology)
    candidate_count = len(topology["candidate_nodes"])
    top1 = top3 = top5 = within1 = within2 = 0
    reciprocal = []
    hop_distances = []
    length_distances = []
    rows = []

    for prediction in predictions:
        truth = by_event[prediction["event_id"]]["node_id"]
        ranking = [row["node_id"] for row in prediction["ranked_nodes"]]
        if truth not in ranking:
            raise RuntimeError(f"True node {truth} missing from candidate ranking")
        rank = ranking.index(truth) + 1
        best = ranking[0]
        hops = shortest(g, best, truth, weighted=False)
        length = shortest(g, best, truth, weighted=True)
        top1 += rank <= 1
        top3 += rank <= 3
        top5 += rank <= 5
        reciprocal.append(1.0 / rank)
        if hops is not None:
            hop_distances.append(hops)
            within1 += hops <= 1
            within2 += hops <= 2
        if length is not None:
            length_distances.append(length)
        rows.append({
            "event_id": prediction["event_id"],
            "scenario_id": prediction["scenario_id"],
            "true_node": truth,
            "top1_node": best,
            "true_node_rank": rank,
            "top1_hop_distance": hops,
            "top1_network_length_distance": length,
        })

    n = len(rows)
    result = {
        "event_count": n,
        "candidate_node_count": candidate_count,
        "top1_accuracy": top1 / n if n else 0.0,
        "top3_recall": top3 / n if n else 0.0,
        "top5_recall": top5 / n if n else 0.0,
        "mean_reciprocal_rank": sum(reciprocal) / n if n else 0.0,
        "top1_within_1_graph_hop": within1 / n if n else 0.0,
        "top1_within_2_graph_hops": within2 / n if n else 0.0,
        "mean_top1_graph_hops": sum(hop_distances) / len(hop_distances) if hop_distances else None,
        "mean_top1_network_length": sum(length_distances) / len(length_distances) if length_distances else None,
        "search_space_reduction_at_top3": 1.0 - min(3, candidate_count) / candidate_count if candidate_count else 0.0,
        "per_event": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_event"}, indent=2))


if __name__ == "__main__":
    main()
