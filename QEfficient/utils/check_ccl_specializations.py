# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Set, Tuple


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
    """Return True if n is a power of two (n > 0 and n & (n - 1) == 0)."""
    return n > 0 and (n & (n - 1)) == 0


def band_index_from_mapped_cl(mapped_cl: int) -> int:
    """
    Compute band index ∈ {0,1,2} from mapped_cl using bit arithmetic.

    Bands (upper bounds): 2^15=32768 → idx=0,  2^16=65536 → idx=1,  2^17=131072 → idx=2.
    For mapped_cl > 131072, clamp to idx=2.
    """
    # ceil(log2(mapped_cl)) == bit_length(mapped_cl - 1)
    ceil_log2 = (mapped_cl - 1).bit_length()
    # map to {0,1,2} by subtracting 15 (the exponent for 32768) and clamping
    idx = max(0, min(2, ceil_log2 - 15))
    return idx


def build_doubling_set(start: int, limit: int, max_elements: int) -> Set[int]:
    """
    Build a STRICT doubling set: {start, start*2, start*4, ...} up to 'limit',
    collecting at most 'max_elements' values. Returns a set; caller will sort.
    """
    values: Set[int] = set()
    if max_elements <= 0 or start <= 0 or limit <= 0:
        return values

    v = start
    while v <= limit and len(values) < max_elements:
        values.add(v)
        v *= 2
    return values


def ensure_last(sorted_seq: List[int], last_value: int, max_elements: int) -> List[int]:
    """
    Ensure the last element equals 'last_value' by appending or replacing the final element,
    keeping length <= max_elements. If the sequence is empty, return [last_value].
    """
    if max_elements <= 0:
        return []
    if not sorted_seq:
        return [last_value][:max_elements]
    if sorted_seq[-1] != last_value:
        if len(sorted_seq) < max_elements:
            sorted_seq.append(last_value)
        else:
            sorted_seq[-1] = last_value
    return sorted_seq[:max_elements]


def automatic_ccl_generation(
    ctx_len: int,
    prefill_seq_len: int,
    comp_ctx_lengths_prefill: Optional[List[int]] = None,
    comp_ctx_lengths_decode: Optional[List[int]] = None,
) -> Tuple[List[int], List[int], int]:
    """
    Automatic Compute-Context-Length Lists Generation

    Purpose:
        Compute decode and prefill CCL lists based on an input context length (CL),
        prefill sequence length, and optional pre-specified lists.

    High-level rules (unchanged from your finalized logic):
        - prefill_seq_len > 1:
            * If either list is provided, pass them through unchanged.
            * decode: doubles from tiered start; MUST end at mapped_CL (last forced to mapped_CL).
            * prefill:
                • If CL is power of two: STRICT doubling from tiered start, bounded by CL (no forced non-doubling last).
                • Else: doubles from tiered start, bounded by CL, and last element = floor_to_1000(mapped_CL).
            * Max 5 elements per list.
        - prefill_seq_len == 1:
            * decode and prefill are IDENTICAL.
            * start at 4096, double up to 10 elements.
            * upper grid cap computed dynamically (start * 2^(max_elements-1)); last = mapped_CL.
            * If mapped_CL < 4096, both lists are [mapped_CL].
    """
    # Handle non-positive CL
    if ctx_len <= 0:
        mapped_cl = next_multiple_of_1024(1)
        seq = [mapped_cl]
        return seq, seq, mapped_cl

    mapped_cl = next_multiple_of_1024(ctx_len)

    # Early small-ctx_len case for identical lists
    if mapped_cl <= 4096:
        seq = [mapped_cl]
        return seq, seq, mapped_cl

    # Compute tier starts via band index (no hard-coded chain)
    idx = band_index_from_mapped_cl(mapped_cl)
    decode_start = 4096 << idx  # 4096, 8192, 16384
    PREFILL_STARTS = {0: 4000, 1: 8000, 2: 16000}
    prefill_start = PREFILL_STARTS[idx]

    # Branch: prefill_seq_len > 1
    if prefill_seq_len > 1:
        # Passthrough if either provided
        if comp_ctx_lengths_decode is not None or comp_ctx_lengths_prefill is not None:
            return (
                comp_ctx_lengths_prefill if comp_ctx_lengths_prefill is not None else [],
                comp_ctx_lengths_decode if comp_ctx_lengths_decode is not None else [],
                mapped_cl,
            )

        # Due to limitations in the number of specializations during compilation, we set the maximum number of elements in comp_ctx_lengths_decode and comp_ctx_lengths_prefill lists to 5.
        max_elems = 5

        # ---- Decode: strict doubling up to mapped_cl, then enforce last = mapped_cl
        decode_set = build_doubling_set(start=decode_start, limit=mapped_cl, max_elements=max_elems)
        decode_list = sorted(decode_set)
        decode_list = ensure_last(decode_list, last_value=mapped_cl, max_elements=max_elems)

        # ---- Prefill:
        if is_power_of_two(ctx_len):
            # STRICT doubling only, bounded by ctx_len; do NOT force a non-doubling last
            prefill_set = build_doubling_set(start=prefill_start, limit=ctx_len, max_elements=max_elems)
            prefill_list = sorted(prefill_set)[:max_elems]
        else:
            # Doubles bounded by ctx_len, but last must equal floor_to_1000(mapped_cl)
            prefill_last = floor_to_1000(mapped_cl)
            prefill_set = build_doubling_set(start=prefill_start, limit=ctx_len, max_elements=max_elems)
            prefill_list = sorted(prefill_set)
            prefill_list = ensure_last(prefill_list, last_value=prefill_last, max_elements=max_elems)

        # NOTE: return order preserved from your last snippet (prefill first, then decode)
        return prefill_list, decode_list, mapped_cl

    # Branch: prefill_seq_len == 1 → identical lists
    else:
        # When prefill_seq_len=1 such as in MoE models, prefilling and decoding processes can use the same specializations and we can double the length of Ccl lists.
        # Due to limitations in the number of specializations during compilation, we set the maximum number of elements in comp_ctx_lengths_decode and comp_ctx_lengths_prefill lists to 10.
        max_elems = 10
        start_identical = 4096

        if mapped_cl < start_identical:
            seq = [mapped_cl]
            return seq, seq, mapped_cl

        # Dynamic grid cap: start * 2^(max_elems - 1)
        grid_cap = start_identical * (1 << (max_elems - 1))
        limit = min(mapped_cl, grid_cap)

        seq_set = build_doubling_set(start=start_identical, limit=limit, max_elements=max_elems)
        seq_list = sorted(seq_set)
        seq_list = ensure_last(seq_list, last_value=mapped_cl, max_elements=max_elems)

        return seq_list, seq_list, mapped_cl


def process_ccl_specializations(ccl_prefill, ccl_decode, ctx_len, prefill_seq_len):
    # Automatic CCL generation: If both ccl_prefill and ccl_decode are None,
    # generate optimized context length lists for prefill and decode based on ctx_len
    if ccl_prefill is None and ccl_decode is None:
        ccl_prefill, ccl_decode, ctx_len = automatic_ccl_generation(ctx_len, prefill_seq_len, ccl_prefill, ccl_decode)
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
