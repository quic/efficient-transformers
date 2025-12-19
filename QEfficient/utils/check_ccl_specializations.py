# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Tuple

from QEfficient.utils import constants
from QEfficient.utils.logging_utils import logger


# Better performance when context length is multiple of 1024 → map CL to the next multiple of 1024
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


def build_doubling_list(start: int, limit: int, max_elements: int, last_value: int = None) -> List[int]:
    """
    Build a STRICT doubling list: {start, start*2, start*4, ...} up to 'limit',
    collecting at most 'max_elements' values. Returns a list.
    Ensure the last element equals 'last_value' by appending or replacing the final element.
    """
    values: List[int] = []
    if max_elements <= 0 or start <= 0 or limit <= 0:
        return values

    element = start
    while element <= limit and len(values) < max_elements:
        values.append(element)
        element *= 2

    if last_value is not None and values[-1] != last_value:
        if len(values) < max_elements:
            values.append(last_value)
        else:
            values[-1] = last_value
    return values[:max_elements]


def automatic_ccl_generation(
    ctx_len: int,
    prefill_seq_len: int,
) -> Tuple[List[int], List[int], int]:
    """
    Automatic Compute-Context-Length Lists Generation
    Purpose:
        Compute decode and prefill CCL lists based on an input context length (CL),
        prefill sequence length, and optional pre-specified lists.
    """
    # Handle non-positive CL
    if ctx_len <= 0:
        mapped_cl = next_multiple_of_1024(1)
        seq = [mapped_cl]
        return seq, seq, mapped_cl

    mapped_cl = next_multiple_of_1024(ctx_len)

    # Early small-ctx_len case for identical lists
    if mapped_cl <= constants.CCL_START_CTX_LEN:
        seq = [mapped_cl]
        return seq, seq, mapped_cl

    # To limit the number of elements in CCL list, the starting point will be calculated based on context length
    for upper_bound, (decode_start, prefill_start) in constants.CCL_START_MAP.items():
        if mapped_cl <= upper_bound:
            break

    if prefill_seq_len > 1:
        # ---- Decode: strict doubling up to mapped_cl, then enforce last = mapped_cl
        decode_list = build_doubling_list(
            start=decode_start, limit=mapped_cl, max_elements=constants.CCL_MAX_ELEMENTS_LISTS, last_value=mapped_cl
        )

        # ---- Prefill:
        if is_power_of_two(mapped_cl):
            # STRICT doubling only, bounded by mapped_cl
            prefill_list = build_doubling_list(
                start=prefill_start, limit=mapped_cl, max_elements=constants.CCL_MAX_ELEMENTS_LISTS
            )
        else:
            # Doubles bounded by mapped_cl, but last must equal floor_to_1000(mapped_cl)
            prefill_last = floor_to_1000(mapped_cl)
            prefill_list = build_doubling_list(
                start=prefill_start,
                limit=mapped_cl,
                max_elements=constants.CCL_MAX_ELEMENTS_LISTS,
                last_value=prefill_last,
            )

        return prefill_list, decode_list, mapped_cl

    elif prefill_seq_len == 1:
        # When prefill_seq_len=1 such as in MoE models, prefilling and decoding processes can use the same specializations and we can double the length of ccl lists.
        # Due to limitations in the number of specializations during compilation, we set the maximum number of elements in comp_ctx_lengths_decode and comp_ctx_lengths_prefill lists to 2*constants.CCL_MAX_ELEMENTS_LISTS.
        max_elems = 2 * constants.CCL_MAX_ELEMENTS_LISTS

        if mapped_cl < constants.CCL_START_CTX_LEN:
            seq = [mapped_cl]
            return seq, seq, mapped_cl

        limit = min(mapped_cl, constants.CCL_START_CTX_LEN * (2 ** (max_elems - 1)))

        seq_list = build_doubling_list(
            start=constants.CCL_START_CTX_LEN, limit=limit, max_elements=max_elems, last_value=mapped_cl
        )

        return seq_list, seq_list, mapped_cl
    else:
        logger.warning("prefill_seq_len cannot be less than 1!")


def process_ccl_specializations(ccl_prefill, ccl_decode, ctx_len, prefill_seq_len):
    # Automatic CCL generation: If both ccl_prefill and ccl_decode are None
    if ccl_prefill is None and ccl_decode is None:
        # Generate optimized context length lists for prefill and decode based on ctx_len
        # Due to compiler limitations, ccl_prefill and ccl_decode must have distinct values
        ccl_prefill, ccl_decode, ctx_len = automatic_ccl_generation(ctx_len, prefill_seq_len)
    else:
        if prefill_seq_len == 1:
            if ccl_prefill is not None and ccl_decode is not None:
                # both prefill and decode ccl can share the same specializations since prefill_seq_len=1. So, a sorted union of both lists can be used for both of them.
                ccl_union_all = sorted(set([min(x, ctx_len) for x in ccl_prefill + ccl_decode]))
                ccl_prefill = ccl_union_all
                ccl_decode = ccl_union_all
        else:
            if ccl_prefill:
                ccl_prefill = sorted({min(x, ctx_len) for x in (ccl_prefill)})
            if ccl_decode:
                ccl_decode = sorted({min(x, ctx_len) for x in (ccl_decode)})

            if ccl_prefill is not None and ccl_decode is not None:
                tmp_prefill = ccl_prefill
                ccl_prefill = []
                for val in tmp_prefill:
                    while val in ccl_decode or val in ccl_prefill:
                        val -= 1
                        if val < 0:
                            break  # Prevent negative values
                    if val >= 0:
                        ccl_prefill.append(val)
                ccl_prefill.sort()

    logger.info("CCL Configuration:")
    logger.info(f"  - Prefill context lengths: {ccl_prefill}")
    logger.info(f"  - Decode context lengths: {ccl_decode}")
    logger.info(f"  - Max context length: {ctx_len}")
    return ccl_prefill, ccl_decode, ctx_len
