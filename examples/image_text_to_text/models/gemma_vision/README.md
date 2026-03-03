# Gemma3 NPI Files

a) For Gemma3-4B model user is adviced to use the NPI file namely fp32_nodes_gemma3_4b.yaml
    example compile command -
    npi_file_path = "configs/fp32_nodes_gemma3_4b.yaml"
    npi_file_full_path = os.path.join(os.getcwd(), npi_file_path)

    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=3072,
        img_size=896,
        num_cores=16,
        num_devices=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
        node_precision_info=npi_file_full_path
        )

b) For Gemma3-27B model user is adviced to use the NPI file namely gemma_updated_npi.yaml

    example compile command -
    npi_file_path = "configs/gemma_updated_npi.yaml"
    npi_file_full_path = os.path.join(os.getcwd(), npi_file_path)

    qeff_model.compile(
        prefill_seq_len=128,
        ctx_len=3072,
        img_size=896,
        num_cores=16,
        num_devices=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        skip_vision=True,
        mos=1,
        node_precision_info=npi_file_full_path
        )