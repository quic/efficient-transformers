# We should be using disaggragate serving for KImi-K2 model for best performance
 - Kimi-K2 model has 384/8 ratio of total_experts/experts_per_tok
 - Currently We use read all experts only once always strategy in prefill-only model
 - And we treat weights activtions meaning read only chosen experts for decode-only model

# Multi-head Latent Attention(MLA)
Kimi-K2 uses Multi-head Latent Attention(MLA) which is impleneted with dual cache (for compressed_kv and k_pe) 

# Absorption
MLA has 3 configurations based on order of evaluation different matrices, to enable, mla absorption config needs to passed like this :
- No absorption : mla_absorption = {"cache_compressed": True, "absorption": False, "online": False}
- Offline No absorption : mla_absorption = {"cache_compressed": True, "absorption": True, "online": False}
- Online absorption : mla_absorption = {"cache_compressed": True, "absorption": True, "online": True}

mla_absorption has 3 keys: 
- cache_compressed: True/False -> gets enabled if compressed KVs are cached to save memory.
- absorption: True/False -> gets enabled only when compressed cache is used, if True, enables absorption of attention matrices for efficiency.
- online: True/False -> gets enabled only when absorption is True, enables on device absorption.

# Blocking
We have also implemented KV head replication, HEAD Blocking and KV Blocking which can be enable like this : 
- For No Blocking : qaic_config = {"mla_absorption" : mla_absorption}
- For No blocking with kv head replication : qaic_config = {"mla_absorption" : mla_absorption, "num_kv_heads_repeat": TS}
- For KV blocking : qaic_config = {"mla_absorption" : mla_absorption, "enable_blocking": True, "blocking_mode": "kv"}  # for KV blocking
- For Head Blocking : qaic_config = {"mla_absorption" : mla_absorption, "enable_blocking": True, "blocking_mode": "h", "num_kv_heads_repeat": TS} for h blocking, it internally sets head_block_size equal to num_devices/num_kv_heads_repeat

- Currently Decode-Only model is giving best perf with Head Blocking and compressed cache.
- Contnuous batching is not enabled yet.