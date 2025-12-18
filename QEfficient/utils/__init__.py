# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.transformers.quantizers.auto import (  # noqa: F401
    replace_transformers_quantizers,
    undo_transformers_quantizers,
)
from QEfficient.utils._utils import (  # noqa: F401
    LRUCache,
    check_and_assign_cache_dir,
    create_json,
    create_model_params,
    custom_format_warning,
    dump_qconfig,
    generate_mdp_partition_config,
    get_num_layers_from_config,
    get_num_layers_vlm,
    get_onnx_dir_name,
    get_padding_shape_from_config,
    get_padding_shape_vlm,
    get_qpc_dir_path,
    get_sliding_window_layers,
    get_sliding_window_shapes,
    hf_download,
    load_hf_processor,
    load_hf_tokenizer,
    load_json,
    login_and_download_hf_lm,
    make_serializable,
    onnx_exists,
    padding_check_and_fix,
    qpc_exists,
)
from QEfficient.utils.hash_utils import (  # noqa: F401
    create_export_hash,
    hash_dict_params,
)
