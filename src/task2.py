from __future__ import annotations

from collections import deque
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


def building_transactions(df: pd.DataFrame) -> List[Set[str]]:
    """
    Data is grouped by customer and date. Each group collects all items
    bought in that basket. Only baskets with 2 or more items ar kept.
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
    Here the graph is created. In each basket, pairs of items are found
    This greated a weighted graph.
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


def top_copurchaseditems(graph: Graph, item: str, k: int = 5) -> List[Tuple[str, int]]:
    """
    Items most commonly bought amongst other items are now found
    Each item's neighbour in the graph is found, and their following count
    Neighbours are sorted from high to low
    """
    item = item.lower().strip()
    if item not in graph:
        return []

    neighbors = list(graph[item].items())  # (neighbor, weight)
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return neighbors[:k]


def top_bundles(transactions: List[Set[str]], k: int = 3) -> List[Tuple[Tuple[str, str], int]]:
    """
    Here the most common item pairs are found.
    It counts how many times each two item combos appear across all of the baskets
    These two item combos ar then sorted from high to low
    """
    pair_counts: Dict[Tuple[str, str], int] = {}
    for basket in transactions:
        for a, b in combinations(sorted(basket), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    pairs = list(pair_counts.items())  # ((a,b), count)
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def are_itemscopurchased(graph: Graph, item_a: str, item_b: str, min_count: int = 1) -> bool:
    """
    Here it checks whether or not two items were bought together.
    Direct connections are found between the two items in the graph
    It checks whether or not the connection count meets the minimum
    True is only returned if the two items were bought together.
    """
    a = item_a.lower().strip()
    b = item_b.lower().strip()
    return a in graph and b in graph[a] and graph[a][b] >= min_count


def bfs_related_items(graph: Graph, start_item: str, max_depth: int = 2) -> List[str]:
    """
    Here items related to the starting items are found, using the BFS search
    It starts with the chosen item, then explores every connection from it.
    The search then stops once it reaches the end
    It then returns a list to us, of the related items found.
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
    transactions = building_transactions(df)
    graph = build_cooccurrence_graph(transactions)

    print("Rows:", len(df))
    print("Transactions:", len(transactions))
    print("Unique items (graph nodes):", len(graph))

    # Use a valid dataset item to demonstrate non-empty results
    item = "whole milk"

    print(f"\nTop co purchased with '{item}':", top_copurchaseditems(graph, item, k=5))

    print("\nThese are the top 3 bundles:", top_bundles(transactions, k=3))

    print("\nCo-purchased check on whole milk & soda:",
          are_itemscopurchased(graph, "whole milk", "soda"))

    print("\nBFS related items, from whole milk (depth 2) sample:",
          bfs_related_items(graph, "whole milk", max_depth=2)[:10])


if __name__ == "__main__":
    main()
