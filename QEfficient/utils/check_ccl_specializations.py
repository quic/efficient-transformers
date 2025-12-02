# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


def process_ccl_specializations(qaic_config):
    if qaic_config is None:
        return None, None
    ccl_prefill = qaic_config.pop("comp_ctx_lengths_prefill", None)
    ccl_decode = qaic_config.pop("comp_ctx_lengths_decode", None)
    ctx_len = qaic_config.pop("ctx_len", None)
    prefill_seq_len = qaic_config.pop("prefill_seq_len", 128)

    if ccl_prefill is None or ccl_decode is None:
        return None, None

    if ctx_len is None:
        raise TypeError("`ctx_len` is required when loading the model with CCL.")

    if prefill_seq_len == 1:
        # both prefill and decode ccl can share the same specializations since prefill_seq_len=1. So, a sorted union of both lists can be used for both of them.
        ccl_union_all = sorted(set(ccl_prefill + ccl_decode))
        ccl_union_all = [min(x, ctx_len) for x in ccl_union_all]
        return ccl_union_all, ccl_union_all

    # Step 1: Cap values to ctx_len
    ccl_prefill = [min(x, ctx_len) for x in ccl_prefill]
    ccl_decode = [min(x, ctx_len) for x in ccl_decode]

    # Step 2: Remove duplicates within each list
    ccl_prefill = list(set(ccl_prefill))
    ccl_decode = list(set(ccl_decode))

    # Step 3: Ensure no overlap between ccl_prefill and ccl_decode
    updated_prefill = []
    for val in ccl_prefill:
        while val in ccl_decode or val in updated_prefill:
            val -= 1
            if val < 0:
                break  # Prevent negative values
        if val >= 0:
            updated_prefill.append(val)

    # Step 4: Sort both lists
    updated_prefill.sort()
    ccl_decode.sort()

    return updated_prefill, ccl_decode
