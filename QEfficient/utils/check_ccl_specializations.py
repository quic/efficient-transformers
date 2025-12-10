# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple


def next_multiple_of_1024(n: int) -> int:
    """Ceil 'n' to the next multiple of 1024."""
    if n <= 0:
        return 0
    return ((n + 1023) // 1024) * 1024


def floor_to_1000(n: int) -> int:
    """Floor 'n' to the nearest lower multiple of 1000."""
    if n <= 0:
        return 0
    return (n // 1000) * 1000


def is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two (n>0 and n&(n-1)==0)."""
    return n > 0 and (n & (n - 1)) == 0


def build_doubling_sequence(start: int, limit: int, max_elements: int, force_last: Optional[int] = None) -> List[int]:
    """
    Build an increasing sequence starting at 'start', doubling each step,
    not exceeding 'limit', with total length <= max_elements.
    If 'force_last' is provided, ensure the last element equals 'force_last'
    (replacing/appending as needed), even if it exceeds 'limit'.
    """
    if max_elements <= 0:
        return []

    # If start is already beyond limit, return [force_last or limit] as a single element.
    if start > limit:
        seq = [force_last if force_last is not None else limit]
        return seq[:max_elements]

    seq: List[int] = []
    val = start

    while val <= limit and len(seq) < max_elements:
        seq.append(val)
        next_val = val * 2
        if next_val > limit or len(seq) >= max_elements:
            break
        val = next_val

    # Add/replace last element if a 'force_last' is requested
    if force_last is not None:
        if len(seq) == 0:
            seq = [force_last]
        elif seq[-1] != force_last:
            if len(seq) < max_elements:
                seq.append(force_last)
            else:
                seq[-1] = force_last

    # Deduplicate while preserving order
    dedup = []
    seen = set()
    for x in seq:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup[:max_elements]


def Automatic_CCL_Generation(
    CL: int,
    prefill_seq_len: int,
    comp_ctx_lengths_prefill: Optional[List[int]] = None,
    comp_ctx_lengths_decode: Optional[List[int]] = None,
) -> Tuple[List[int], List[int], int]:
    """
    Automatic Compute-Context-Length Lists Generation

    Purpose:
        Compute decode and prefill ccl lists based on an input context
        length (CL), prefill sequence length, and optional pre-specified lists.
    """

    if CL <= 0:
        mapped_CL = next_multiple_of_1024(max(CL, 1))
        # For non-positive CL, minimal identical sequences
        seq = [mapped_CL]
        return seq, seq, mapped_CL

    mapped_CL = next_multiple_of_1024(CL)

    # Tiered starts
    if mapped_CL <= 4096:
        seq = [mapped_CL]
        return seq, seq, mapped_CL
    elif mapped_CL <= 32768:
        decode_start, prefill_start = 4096, 4000
    elif mapped_CL <= 65536:
        decode_start, prefill_start = 8192, 8000
    elif mapped_CL <= 131072:
        decode_start, prefill_start = 16384, 16000
    else:
        decode_start, prefill_start = 16384, 16000

    # If prefill_seq_len > 1:
    if prefill_seq_len > 1:
        # Passthrough if either provided
        if comp_ctx_lengths_decode is not None or comp_ctx_lengths_prefill is not None:
            return (
                comp_ctx_lengths_decode if comp_ctx_lengths_decode is not None else [],
                comp_ctx_lengths_prefill if comp_ctx_lengths_prefill is not None else [],
                mapped_CL,
            )

        max_elems = 5

        # Decode: ensure last = mapped_CL
        decode = build_doubling_sequence(
            start=decode_start,
            limit=mapped_CL,
            max_elements=max_elems,
            force_last=mapped_CL,
        )

        # Prefill:
        if is_power_of_two(CL):
            # Strict doubling, limit = CL, no forced non-doubling last
            prefill = build_doubling_sequence(
                start=prefill_start,
                limit=CL,
                max_elements=max_elems,
                force_last=None,
            )
        else:
            prefill_last = floor_to_1000(mapped_CL)
            prefill = build_doubling_sequence(
                start=prefill_start,
                limit=CL,
                max_elements=max_elems,
                force_last=prefill_last,
            )

        return prefill, decode, mapped_CL

    # UPDATED: prefill_seq_len == 1 → identical lists
    else:
        max_elems = 10
        grid_cap = 2097152  # upper cap for doubling grid

        if mapped_CL < 4096:
            seq = [mapped_CL]
        else:
            seq = build_doubling_sequence(
                start=4096,
                limit=min(mapped_CL, grid_cap),
                max_elements=max_elems,
                force_last=mapped_CL,  # identical lists end at mapped_CL
            )
        return seq, seq, mapped_CL


def process_ccl_specializations(ccl_prefill, ccl_decode, ctx_len, prefill_seq_len):
    # Automatic CCL generation: If both ccl_prefill and ccl_decode are None,
    # generate optimized context length lists for prefill and decode based on ctx_len
    if ccl_prefill is None and ccl_decode is None:
        ccl_prefill, ccl_decode, ctx_len = Automatic_CCL_Generation(ctx_len, prefill_seq_len, ccl_prefill, ccl_decode)
    else:
        if prefill_seq_len == 1:
            if ccl_prefill is not None and ccl_decode is not None:
                # both prefill and decode ccl can share the same specializations since prefill_seq_len=1. So, a sorted union of both lists can be used for both of them.
                ccl_union_all = sorted(set(ccl_prefill + ccl_decode))
                ccl_union_all = [min(x, ctx_len) for x in ccl_union_all]
                ccl_prefill = ccl_union_all
                ccl_decode = ccl_union_all
        else:
            # Step 1: Cap values to ctx_len
            ccl_prefill = [min(x, ctx_len) for x in ccl_prefill] if ccl_prefill is not None else None
            ccl_decode = [min(x, ctx_len) for x in ccl_decode] if ccl_decode is not None else None

            # Step 2: Remove duplicates within each list
            ccl_prefill = list(set(ccl_prefill)) if ccl_prefill is not None else None
            ccl_decode = list(set(ccl_decode)) if ccl_decode is not None else None

            if ccl_prefill is None or ccl_decode is None:
                if ccl_prefill:
                    ccl_prefill.sort()
                if ccl_decode:
                    ccl_decode.sort()
            else:
                # Step 3: Ensure no overlap between ccl_prefill and ccl_decode
                tmp_prefill = ccl_prefill
                ccl_prefill = []
                for val in tmp_prefill:
                    while val in ccl_decode or val in ccl_prefill:
                        val -= 1
                        if val < 0:
                            break  # Prevent negative values
                    if val >= 0:
                        ccl_prefill.append(val)

                # Step 4: Sort both lists
                ccl_prefill.sort()
                ccl_decode.sort()

    print("CCL Configuration:")
    print(f"  - Prefill context lengths: {ccl_prefill}")
    print(f"  - Decode context lengths: {ccl_decode}")
    print(f"  - Max context length: {ctx_len}")
    return ccl_prefill, ccl_decode, ctx_len
