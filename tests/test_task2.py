import pandas as pd

from src.task2 import (
    build_transactions,
    build_cooccurrence_graph,
    top_copurchased,
    top_bundles,
    are_copurchased,
    bfs_related_items,
)

def test_build_transactions_groups_member_date():
    df = pd.DataFrame({
        "Member_number": [1, 1, 1, 2],
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01"],
        "itemDescription": ["bread", "milk", "eggs", "bread"]
    })
    tx = build_transactions(df)
    # transactions with >=2 items only => (1, 2024-01-01) has 2 items; others ignored
    assert len(tx) == 1
    assert tx[0] == {"bread", "milk"}

def test_graph_weights_increment():
    transactions = [{"bread", "milk"}, {"bread", "milk"}]
    g = build_cooccurrence_graph(transactions)
    assert g["bread"]["milk"] == 2
    assert g["milk"]["bread"] == 2

def test_top_copurchased_empty_for_unknown_item():
    g = {"bread": {"milk": 3}}
    assert top_copurchased(g, "unknown", 5) == []

def test_top_bundles_counts_pairs():
    transactions = [{"bread", "milk"}, {"bread", "milk"}, {"bread", "eggs"}]
    top = top_bundles(transactions, k=1)
    assert top[0][0] == ("bread", "milk")
    assert top[0][1] == 2

def test_are_copurchased_threshold():
    g = {"bread": {"milk": 2}, "milk": {"bread": 2}}
    assert are_copurchased(g, "bread", "milk", min_count=2) is True
    assert are_copurchased(g, "bread", "milk", min_count=3) is False

def test_bfs_related_items_depth():
    g = {
        "bread": {"milk": 1},
        "milk": {"bread": 1, "eggs": 1},
        "eggs": {"milk": 1},
    }
    found = bfs_related_items(g, "bread", max_depth=1)
    assert "milk" in found
    assert "eggs" not in found
