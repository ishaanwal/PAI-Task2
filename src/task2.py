from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Supermarket_dataset_PAI.csv"


# ----------------------------
# Data model
# ----------------------------
@dataclass(frozen=True)
class Purchase:
    member_number: int
    date: str            # keep as string for simplicity unless brief demands datetime
    item: str


# ----------------------------
# Load + clean
# ----------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    df = df.dropna(subset=["Member_number", "Date", "itemDescription"]).copy()
    df["Member_number"] = df["Member_number"].astype(int)
    df["Date"] = df["Date"].astype(str).str.strip()
    df["itemDescription"] = df["itemDescription"].astype(str).str.strip().str.lower()

    return df


def to_purchases(df: pd.DataFrame) -> List[Purchase]:
    return [
        Purchase(int(row.Member_number), str(row.Date), str(row.itemDescription))
        for row in df.itertuples(index=False)
    ]


# ----------------------------
# 1) Searching
# ----------------------------
def linear_search_member(purchases: List[Purchase], target_member: int) -> List[Purchase]:
    """Return all purchases for a member using linear scan. O(n)."""
    results = []
    for p in purchases:
        if p.member_number == target_member:
            results.append(p)
    return results


def merge_sort_purchases_by_member(purchases: List[Purchase]) -> List[Purchase]:
    """Merge sort purchases by member_number. O(n log n)."""

    if len(purchases) <= 1:
        return purchases

    mid = len(purchases) // 2
    left = merge_sort_purchases_by_member(purchases[:mid])
    right = merge_sort_purchases_by_member(purchases[mid:])

    return _merge_by_member(left, right)


def _merge_by_member(left: List[Purchase], right: List[Purchase]) -> List[Purchase]:
    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i].member_number <= right[j].member_number:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def binary_search_member_range(sorted_purchases: List[Purchase], target_member: int) -> Tuple[int, int]:
    """
    Return (start_index, end_index_exclusive) for a member in a list sorted by member_number.
    If not found, returns (-1, -1).
    Overall: O(log n) to locate boundaries.
    """
    left = _lower_bound_member(sorted_purchases, target_member)
    if left == len(sorted_purchases) or sorted_purchases[left].member_number != target_member:
        return -1, -1
    right = _upper_bound_member(sorted_purchases, target_member)
    return left, right


def _lower_bound_member(arr: List[Purchase], x: int) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid].member_number < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _upper_bound_member(arr: List[Purchase], x: int) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid].member_number <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ----------------------------
# 2) Frequency counting + sorting items (uses merge sort again)
# ----------------------------
def item_frequencies(purchases: List[Purchase]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for p in purchases:
        freq[p.item] = freq.get(p.item, 0) + 1
    return freq


def merge_sort_items_by_count(items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Sort (item, count) descending by count using merge sort. O(m log m)."""
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left = merge_sort_items_by_count(items[:mid])
    right = merge_sort_items_by_count(items[mid:])
    return _merge_items_by_count(left, right)


def _merge_items_by_count(left: List[Tuple[str, int]], right: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        # DESC by count
        if left[i][1] >= right[j][1]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


# ----------------------------
# Outputs (what you’ll paste into the report)
# ----------------------------
def main():
    df = load_data()
    purchases = to_purchases(df)

    print("Dataset loaded successfully")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print()

    # Frequency output
    freq = item_frequencies(purchases)
    items_sorted = merge_sort_items_by_count(list(freq.items()))
    top10 = items_sorted[:10]

    print("Top 10 most frequent items:")
    for item, count in top10:
        print(f"- {item}: {count}")
    print()

    # Demo search (pick any member number that exists)
    example_member = purchases[0].member_number

    # Linear search
    lin = linear_search_member(purchases, example_member)
    print(f"Linear search: member {example_member} purchases = {len(lin)}")

    # Binary search (requires sort first)
    sorted_by_member = merge_sort_purchases_by_member(purchases)
    start, end = binary_search_member_range(sorted_by_member, example_member)
    if start == -1:
        print(f"Binary search: member {example_member} not found")
    else:
        print(f"Binary search: member {example_member} purchases = {end - start}")
        # show first 5 purchases for that member
        preview = sorted_by_member[start:min(end, start + 5)]
        print("First few purchases for that member:")
        for p in preview:
            print(f"  {p.member_number} | {p.date} | {p.item}")


if __name__ == "__main__":
    main()
