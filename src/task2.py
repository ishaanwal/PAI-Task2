from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Supermarket_dataset_PAI.csv"


# Graph: adjacency[item_a][item_b] = weight (co-purchase frequency)
Graph = Dict[str, Dict[str, int]]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Member_number", "Date", "itemDescription"]).copy()
    df["Member_number"] = df["Member_number"].astype(int)
    df["Date"] = df["Date"].astype(str).str.strip()
    df["itemDescription"] = df["itemDescription"].astype(str).str.strip().str.lower()
    return df


def build_transactions(df: pd.DataFrame) -> List[Set[str]]:
    """
    Each transaction = items bought together in one shopping visit.
    We approximate a visit by grouping on (Member_number, Date).
    """
    transactions: List[Set[str]] = []
    grouped = df.groupby(["Member_number", "Date"])["itemDescription"]

    for _, items in grouped:
        basket = set(items.tolist())
        if len(basket) >= 2:
            transactions.append(basket)

    return transactions


def build_cooccurrence_graph(transactions: List[Set[str]]) -> Graph:
    """
    Build an undirected weighted graph using adjacency dictionaries.
    """
    graph: Graph = {}

    for basket in transactions:
        for a, b in combinations(sorted(basket), 2):
            _add_edge(graph, a, b, 1)
            _add_edge(graph, b, a, 1)

    return graph


def _add_edge(graph: Graph, a: str, b: str, w: int) -> None:
    if a not in graph:
        graph[a] = {}
    graph[a][b] = graph[a].get(b, 0) + w


def top_copurchased(graph: Graph, item: str, k: int = 5) -> List[Tuple[str, int]]:
    """
    Return top-k items most frequently co-purchased with `item`.
    """
    item = item.lower().strip()
    if item not in graph:
        return []

    neighbors = list(graph[item].items())  # (neighbor, weight)
    # sort descending by weight
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:k]


def top_bundles(transactions: List[Set[str]], k: int = 3) -> List[Tuple[Tuple[str, str], int]]:
    """
    Top-k most common 2-item bundles across all transactions.
    """
    pair_counts: Dict[Tuple[str, str], int] = {}
    for basket in transactions:
        for a, b in combinations(sorted(basket), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    pairs = list(pair_counts.items())  # ((a,b), count)
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def are_copurchased(graph: Graph, item_a: str, item_b: str, min_count: int = 1) -> bool:
    """
    Quick check: have these two items ever been co-purchased (or >= min_count times)?
    """
    a = item_a.lower().strip()
    b = item_b.lower().strip()
    return a in graph and b in graph[a] and graph[a][b] >= min_count


def bfs_related_items(graph: Graph, start_item: str, max_depth: int = 2) -> List[str]:
    """
    BFS traversal to find items related to `start_item` within `max_depth`.
    Returns discovered items (excluding the start).
    """
    start = start_item.lower().strip()
    if start not in graph:
        return []

    visited = {start}
    q = deque([(start, 0)])
    found: List[str] = []

    while q:
        node, depth = q.popleft()
        if depth == max_depth:
            continue

        for nbr in graph.get(node, {}):
            if nbr not in visited:
                visited.add(nbr)
                found.append(nbr)
                q.append((nbr, depth + 1))

    return found


def main():
    df = load_data()
    transactions = build_transactions(df)
    graph = build_cooccurrence_graph(transactions)

    print("Rows:", len(df))
    print("Transactions:", len(transactions))
    print("Unique items (graph nodes):", len(graph))

    # Example queries (you can change the item to 'bread' etc.)
    item = "bread"
    print(f"\nTop co-purchased with '{item}':", top_copurchased(graph, item, k=5))

    print("\nTop 3 bundles:", top_bundles(transactions, k=3))

    print("\nCo-purchased check bread & milk:", are_copurchased(graph, "bread", "milk"))

    print("\nBFS related items from bread (depth 2) sample:", bfs_related_items(graph, "bread", max_depth=2)[:10])


if __name__ == "__main__":
    main()
