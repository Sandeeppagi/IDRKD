"""Week 3 graph analytics primitives."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyticsResult:
    pagerank: dict[str, float]
    communities: dict[str, int]


def bfs_neighbors(edges: list[tuple[str, str]], start: str, *, depth: int = 2) -> list[str]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    result: list[str] = []
    while queue:
        node, distance = queue.popleft()
        if distance == depth:
            continue
        for neighbor in sorted(adjacency[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                result.append(neighbor)
                queue.append((neighbor, distance + 1))
    return result


def pagerank(edges: list[tuple[str, str]], *, iterations: int = 20, damping: float = 0.85) -> dict[str, float]:
    nodes = sorted({node for edge in edges for node in edge})
    if not nodes:
        return {}
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        outgoing[source].add(target)
    scores = {node: 1.0 / len(nodes) for node in nodes}
    for _ in range(iterations):
        next_scores = {node: (1.0 - damping) / len(nodes) for node in nodes}
        for source, targets in outgoing.items():
            if not targets:
                continue
            share = scores[source] / len(targets)
            for target in targets:
                next_scores[target] += damping * share
        scores = next_scores
    return scores


def louvain_fallback(edges: list[tuple[str, str]]) -> dict[str, int]:
    nodes = sorted({node for edge in edges for node in edge})
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    communities: dict[str, int] = {}
    community_id = 0
    for node in nodes:
        if node in communities:
            continue
        queue = deque([node])
        while queue:
            current = queue.popleft()
            if current in communities:
                continue
            communities[current] = community_id
            queue.extend(sorted(adjacency[current] - communities.keys()))
        community_id += 1
    return communities


class NightlyGraphAnalyticsJob:
    def run(self, edges: list[tuple[str, str]]) -> AnalyticsResult:
        return AnalyticsResult(pagerank=pagerank(edges), communities=louvain_fallback(edges))
