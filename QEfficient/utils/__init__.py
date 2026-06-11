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
    align_kv_input_names_to_retained_outputs,
    apply_kv_cache_prefix,
    check_and_assign_cache_dir,
    create_json,
    create_model_params,
    custom_format_warning,
    dump_qconfig,
    generate_mdp_partition_config,
    get_attr_or_key,
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
    require_value,
    to_named_specializations,
    validate_kv_cache_prefix,
)
from QEfficient.utils.compile_layerwise import (  # noqa: F401
    run_compile_layerwise,
)
from QEfficient.utils.hash_utils import (  # noqa: F401
    create_export_hash,
    hash_dict_params,
)
from QEfficient.utils.layer_scale_checkpoint import (  # noqa: F401
    SUPPORTED_RUNTIME_EQUIVALENCE_MODES,
    LayerScaleRecipe,
    TensorScaleSpec,
    TensorScalingRule,
    apply_layer_scale_recipe_to_loaded_model,
    apply_layer_scale_recipe_to_snapshot,
    build_layer_scale_recipe_from_recovery_json,
    build_layer_scale_recipe_from_recovery_result,
    build_tensor_scale_specs,
    dump_layer_scale_recipe_yaml,
    load_layer_scale_recipe,
    serialize_layer_scale_recipe,
)
from QEfficient.utils.layerwise_pipeline import (  # noqa: F401
    layerwise_pipeline,
)
from QEfficient.utils.precision_recovery import (  # noqa: F401
    DEFAULT_ITERATIVE_MLP_CANDIDATES,
    PrecisionRecoveryRequest,
    detect_precision_recovery_backend,
    run_precision_recovery,
    summarize_precision_recovery_result,
)
from QEfficient.utils.precision_recovery_agent import (  # noqa: F401
    DEFAULT_SCALE_CANDIDATE_SCHEDULES,
    PrecisionRecoveryAgent,
    PrecisionRecoveryAgentRequest,
    needs_scale_search,
    parse_scale_candidate_schedules,
    resolve_model_id_from_card,
    run_precision_recovery_agent,
)
