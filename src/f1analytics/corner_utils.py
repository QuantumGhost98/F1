"""
Shared corner resolution and labeling utilities.

Extracted from CornerAnalysis, CornerSpeedComparator, CornerMinSpeed,
and CornerMinThrottle which all had identical copies of these functions.
"""
import re
from typing import List, Tuple, Union

import pandas as pd


def corner_identifier_to_index(circuit_info, label: Union[int, str]) -> int:
    """
    Resolve a corner identifier like 3, "3", "1A", "12B" to 
    the DataFrame index in circuit_info.corners.

    Handles both zero-based and one-based corner numbering schemes.
    """
    corner_df = circuit_info.corners
    zero_based_numbering = corner_df['Number'].min() == 0

    s = str(label).strip()
    m = re.fullmatch(r'(\d+)([A-Za-z]?)', s)
    if not m:
        raise ValueError(
            f"Invalid corner identifier '{label}' "
            f"(expected digits with optional letter, e.g. '1', '1A')"
        )

    display_num = int(m.group(1))
    letter = m.group(2).upper()

    internal_number = display_num - 1 if zero_based_numbering else display_num

    subset = corner_df[corner_df['Number'] == internal_number]
    if subset.empty:
        raise ValueError(f"Corner number '{display_num}' not found in circuit_info.corners")

    if letter:
        mask = subset['Letter'].astype(str).str.upper() == letter
        subset = subset[mask]
        if subset.empty:
            raise ValueError(
                f"Corner '{label}' not found (number {display_num} with letter '{letter}')"
            )
    else:
        no_letter_mask = subset['Letter'].isna() | (subset['Letter'].astype(str).str.strip() == '')
        filtered = subset[no_letter_mask]
        if not filtered.empty:
            subset = filtered

    label_idx = subset.index[0]
    pos = int(corner_df.index.get_loc(label_idx))
    return pos


def corner_label(circuit_info, apex_idx: int) -> str:
    """
    Return combined Number+Letter label for a given corner index.
    Handles zero-based corner numbering transparently.
    """
    corner_df = circuit_info.corners
    zero_based = corner_df['Number'].min() == 0

    try:
        if apex_idx in corner_df.index:
            row = corner_df.loc[apex_idx]
        else:
            row = corner_df.iloc[apex_idx]
    except Exception:
        row = corner_df.iloc[apex_idx]

    num = int(row['Number']) + (1 if zero_based else 0)
    letter = ''
    if ('Letter' in corner_df.columns
            and pd.notna(row.get('Letter', None))
            and str(row['Letter']).strip()):
        letter = str(row['Letter']).strip()
    return f"{num}{letter}"


def indices_between(a_idx: int, b_idx: int) -> List[int]:
    """Inclusive list of indices between two corner indices (order-agnostic)."""
    start, end = sorted((int(a_idx), int(b_idx)))
    return list(range(start, end + 1))


def compress_indices_to_ranges(idx_list: List[int]) -> List[Tuple[int, int]]:
    """Compress sorted integer indices into (start, end) tuples for contiguous runs."""
    if not idx_list:
        return []
    sorted_idxs = sorted(idx_list)
    ranges: List[Tuple[int, int]] = []
    run_start = prev = sorted_idxs[0]
    for x in sorted_idxs[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append((run_start, prev))
        run_start = prev = x
    ranges.append((run_start, prev))
    return ranges


def format_corner_label_list(circuit_info, idx_list: List[int]) -> str:
    """Return a compact label like '1,3-4,6' from a list of corner apex indices."""
    parts = []
    for a, b in compress_indices_to_ranges(idx_list):
        if a == b:
            parts.append(corner_label(circuit_info, a))
        else:
            parts.append(f"{corner_label(circuit_info, a)}-{corner_label(circuit_info, b)}")
    return ",".join(parts)


def resolve_corner_idxs(circuit_info, corner_idxs) -> List[int]:
    """
    Normalize user-provided corner_idxs into a ordered, deduplicated list.
    
    Accepts:
      - single int or string label (e.g., 3, "1A")
      - iterable of items which can be ints/strings, two-length ranges, 
        or "3-5" style range strings
    """
    def resolve_single(ci):
        return corner_identifier_to_index(circuit_info, ci)

    def expand_item(item):
        # Range as string: "3-5" or "1A-2B"
        if isinstance(item, str) and '-' in item:
            left, right = [p.strip() for p in item.split('-', 1)]
            a = resolve_single(left)
            b = resolve_single(right)
            return indices_between(a, b)

        # Simple scalar
        if isinstance(item, (int, str)):
            return [resolve_single(item)]

        # Two-length iterable = range, e.g. [3, 4] or ("3", "4B")
        if hasattr(item, '__iter__') and not isinstance(item, (bytes, bytearray)):
            try:
                it = list(item)
            except Exception:
                raise ValueError(f"Invalid corner item: {item}")
            if len(it) == 2 and all(isinstance(x, (int, str)) for x in it):
                a = resolve_single(it[0])
                b = resolve_single(it[1])
                return indices_between(a, b)
            # Nested iterables: flatten
            expanded = []
            for sub in it:
                expanded.extend(expand_item(sub))
            return expanded

        raise ValueError(f"Invalid corner item: {item}")

    # Build list
    if isinstance(corner_idxs, (int, str)):
        raw_indices = expand_item(corner_idxs)
    elif hasattr(corner_idxs, '__iter__'):
        raw_indices = []
        for it in corner_idxs:
            raw_indices.extend(expand_item(it))
    else:
        raise ValueError("corner_idxs must be an int, string like '1A', or iterable thereof")

    # Deduplicate preserving order
    seen = set()
    result = []
    for idx in raw_indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)

    if not result:
        raise ValueError("No valid corners resolved from corner_idxs")

    return result
