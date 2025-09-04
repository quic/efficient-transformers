
import onnxscript
import torch
import torch.nn as nn
import math

CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
# Import the ONNX Script opset for version 13
ops = getattr(onnxscript, "opset" + str(13))


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def SD35AdaLayerNormZeroX(
    hidden_states: onnxscript.FLOAT,
    emb: onnxscript.FLOAT, 
    linear_weight: onnxscript.FLOAT,
    linear_bias: onnxscript.FLOAT,
    norm_epsilon: float, 
):
   
    # 1. emb = self.linear(self.silu(emb))
    silu_emb = ops.Mul(emb, ops.Sigmoid(emb)) 
    linear_out = ops.MatMul(silu_emb, ops.Transpose(linear_weight, perm=[1, 0])) 
    linear_out = ops.Add(linear_out, linear_bias)

    # 2. Chunk `linear_out` into 9
    # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2
    # Determine chunk size dynamically, assuming equal chunks.
    output_dim_linear = ops.Shape(linear_out)[-1]
    chunk_size = ops.Cast(output_dim_linear / 9, to=6) # Cast to Int64

    split_sizes = ops.Constant(value_ints=[chunk_size] * 9) # A tuple of 9 chunk_size values
    split_outputs = ops.Split(linear_out, split_size=split_sizes, axis=1)

    shift_msa = split_outputs[0]
    scale_msa = split_outputs[1]
    gate_msa = split_outputs[2]
    shift_mlp = split_outputs[3]
    scale_mlp = split_outputs[4]
    gate_mlp = split_outputs[5]
    shift_msa2 = split_outputs[6]
    scale_msa2 = split_outputs[7]
    gate_msa2 = split_outputs[8]

    # 3. norm_hidden_states = self.norm(hidden_states)
    norm_hidden_states = ops.LayerNormalization(
        hidden_states,
        scale=ops.Constant(value_float=1.0, output_dtype=1), # float
        bias=ops.Constant(value_float=0.0, output_dtype=1), # float
        epsilon=norm_epsilon
    )

    # 4. hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
    # This `hidden_states` becomes the first output of the function.
    output_hidden_states = ops.Add(
        ops.Mul(norm_hidden_states, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_msa, axes=[1]))),
        ops.Unsqueeze(shift_msa, axes=[1])
    )

    # 5. norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
    output_norm_hidden_states2 = ops.Add(
        ops.Mul(norm_hidden_states, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_msa2, axes=[1]))),
        ops.Unsqueeze(shift_msa2, axes=[1])
         )

    # Return signature of SD35AdaLayerNormZeroX's forward is:
    # Tuple[torch.Tensor, ...]: hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2
    return output_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, output_norm_hidden_states2, gate_msa2


@onnxscript.script(CUSTOM_OPSET)
def AdaLayerNormZero(
    x: onnxscript.FLOAT, 
    emb: onnxscript.FLOAT, 

    linear_weight: onnxscript.FLOAT, # Weight for self.linear
    linear_bias: onnxscript.FLOAT,   # Bias for self.linear

    norm_epsilon: float,             
):
    

    # 1. `emb = self.linear(self.silu(emb))`
    silu_emb = ops.Mul(emb, ops.Sigmoid(emb)) # Equivalent to nn.SiLU()

   
    linear_out = ops.MatMul(silu_emb, ops.Transpose(linear_weight, perm=[1, 0]))
    linear_out = ops.Add(linear_out, linear_bias)

    # 2. `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)`
    # The linear_out has a shape of [..., 6 * embedding_dim]
    output_dim_linear = ops.Shape(linear_out)[-1]
    chunk_size = ops.Cast(ops.Div(output_dim_linear, ops.Constant(value_int=6)), to=6) # Cast to Int64

    # ops.Split requires explicit sizes for each chunk
    split_sizes = ops.Constant(value_ints=[chunk_size] * 6)
    split_outputs = ops.Split(linear_out, split_size=split_sizes, axis=1)

    shift_msa = split_outputs[0]
    scale_msa = split_outputs[1]
    gate_msa = split_outputs[2]
    shift_mlp = split_outputs[3]
    scale_mlp = split_outputs[4]
    gate_mlp = split_outputs[5]

    # 3. `x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]`
    norm_x = ops.LayerNormalization(
        x,
        scale=ops.Constant(value_float=1.0, output_dtype=1), # float type (ONNX FLOAT)
        bias=ops.Constant(value_float=0.0, output_dtype=1),  # float type (ONNX FLOAT)
        epsilon=norm_epsilon
    )

    # Apply the scaling and shifting: `norm_x * (1 + scale_msa[:, None]) + shift_msa[:, None]`
    # Use ops.Unsqueeze for `[:, None]` 
    scaled_shifted_x = ops.Add(
        ops.Mul(norm_x, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_msa, axes=[1]))),
        ops.Unsqueeze(shift_msa, axes=[1])
    )

    return scaled_shifted_x, gate_msa, shift_mlp, scale_mlp, gate_mlp

@onnxscript.script(CUSTOM_OPSET)
def GELU(
    hidden_states: onnxscript.FLOAT,
    proj_weight: onnxscript.FLOAT,
    proj_bias: onnxscript.FLOAT,
):
    """
    ONNXScript equivalent of GELU with approximate="tanh" activation.
    Corresponds to:
    hidden_states = nn.Linear(in_dim, out_dim)(hidden_states)
    return 0.5 * hidden_states * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / math.pi)) * (hidden_states + 0.044715 * torch.pow(hidden_states, 3))))
    """
    projected_states = ops.MatMul(hidden_states, ops.Transpose(proj_weight, perm=[1, 0]))
    projected_states = ops.Add(projected_states, proj_bias)

    x = projected_states
    x_cubed = ops.Pow(x, ops.Constant(value_float=3.0))
    term_x_plus_044715_x_cubed = ops.Add(x, ops.Mul(ops.Constant(value_float=0.044715), x_cubed))
    sqrt_2_div_pi = ops.Constant(value_float=math.sqrt(2.0 / math.pi))
    argument_for_tanh = ops.Mul(sqrt_2_div_pi, term_x_plus_044715_x_cubed)
    tanh_val = ops.Tanh(argument_for_tanh)
    one_plus_tanh_val = ops.Add(ops.Constant(value_float=1.0), tanh_val)
    final_gelu_output = ops.Mul(ops.Mul(ops.Constant(value_float=0.5), x), one_plus_tanh_val)

    return final_gelu_output

# ff_context #ff
@onnxscript.script(CUSTOM_OPSET)
def FeedForward(
    hidden_states: onnxscript.FLOAT,
    dim: int,           
    dim_out: int,       # `dim_out` from FeedForward init (output dimension of the block)
    mult: int,          # `mult` from FeedForward init (used to calculate inner_dim)
    dropout_ratio: float, # `dropout` for nn.Dropout
    final_dropout: bool, # `final_dropout` bool
    act_fn_proj_weight: onnxscript.FLOAT, 
    act_fn_proj_bias: onnxscript.FLOAT,   
    project_out_weight: onnxscript.FLOAT,
    project_out_bias: onnxscript.FLOAT,
):
    # Calculate inner_dim as in PyTorch FeedForward.__init__
    # inner_dim = int(dim * mult)
    inner_dim_val = ops.Cast(ops.Mul(dim, mult), to=6) # 6 is ONNX INT64
    # 1. Apply act_fn (which is GELUOnnx here)
    ff_output = GELU(
        hidden_states,
        act_fn_proj_weight,
        act_fn_proj_bias,
    )

    # 2. Apply first Dropout
    
    ff_output = ops.Dropout(ff_output, ratio=dropout_ratio)

    # 3. Apply project out (final Linear layer)
    
    ff_output = ops.MatMul(ff_output, ops.Transpose(project_out_weight, perm=[1, 0]))
    ff_output = ops.Add(ff_output, project_out_bias)

    # 4. Apply final Dropout (if final_dropout is True)
    if final_dropout:
        ff_output = ops.Dropout(ff_output, ratio=dropout_ratio) 

    return ff_output



# qk_norm == "rms_norm"
#recheck this with the customRMSNorm we have earlier
import onnxscript
from onnxscript import opset16 as ops # Using opset16, adjust if needed
from onnxscript.values import Opset

# Define your ONNXScript Opset domain
CUSTOM_OPSET = Opset(domain="com.qualcomm.cloud", version=1)

@onnxscript.script(CUSTOM_OPSET)
def RMSNorm(
    hidden_states: onnxscript.FLOAT,  # Input tensor
    weight: onnxscript.FLOAT,         # Corresponds to self.weight (nn.Parameter)
                                      # Pass an empty tensor or zero tensor if elementwise_affine is False
    eps: float,                       # Corresponds to self.eps
    elementwise_affine: bool,         # Corresponds to elementwise_affine in __init__
):
    """
    ONNXScript equivalent of RMSNorm.
    Handles dtype conversions and conditional weight application.
    """



    # 1. variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
   
    hidden_states_fp32 = ops.Cast(hidden_states, to=1) # 1 is ONNX FLOAT (float32)

    variance = ops.ReduceMean(ops.Pow(hidden_states_fp32, 2), axes=[-1], keepdims=1)

    # 2. hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
  
    variance_with_eps = ops.Add(variance, ops.Constant(value_float=eps))
    rsqrt_val = ops.Reciprocal(ops.Sqrt(variance_with_eps))

   
    hidden_states_normalized = ops.Mul(hidden_states, rsqrt_val)


    # 3. Conditional weight application: if self.weight is not None: hidden_states = hidden_states * self.weight
    # This `if` corresponds to `elementwise_affine` boolean
    if elementwise_affine:
        
        hidden_states_to_weight_dtype = ops.Cast(hidden_states_normalized, to=ops.DTYPE_MAP[weight.dtype]) # type: ignore
        
        output = ops.Mul(hidden_states_to_weight_dtype, weight)
    else:
        output = hidden_states_normalized

    
    
    return output


@onnxscript.script(CUSTOM_OPSET)
def JointAttnProcessor2_0(
    hidden_states: onnxscript.FLOAT,
    encoder_hidden_states: onnxscript.FLOAT,  # This can conceptually be an empty tensor for None
    attention_mask: onnxscript.FLOAT,

    # Parameters from `Attention` module
    attn_heads: int,
    attn_head_dim: int,
    attn_scale_qk: bool,
    attn_scale_val: float, # self.scale
    attn_query_dim: int, # self.query_dim (input dim to to_q, to_out[0])
    attn_inner_dim: int, # output dim of to_q
    attn_inner_kv_dim: int, # output dim of to_k, to_v

    # Weights and Biases for Attention Projections (`attn.to_q`, `attn.to_k`, `attn.to_v`)
    to_q_weight: onnxscript.FLOAT,
    to_q_bias: onnxscript.FLOAT,
    to_k_weight: onnxscript.FLOAT,
    to_k_bias: onnxscript.FLOAT,
    to_v_weight: onnxscript.FLOAT,
    to_v_bias: onnxscript.FLOAT,

    # RMSNorm parameters (`attn.norm_q`, `attn.norm_k`, `attn.norm_added_q`, `attn.norm_added_k`)
    # Pass weight and epsilon for each
    norm_q_weight: onnxscript.FLOAT,
    norm_q_eps: float,
    norm_q_elementwise_affine: bool, # From RMSNorm init

    norm_k_weight: onnxscript.FLOAT,
    norm_k_eps: float,
    norm_k_elementwise_affine: bool,

    # Weights and Biases for Cross-Attention Projections (`attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`)
    add_q_proj_weight: onnxscript.FLOAT,
    add_q_proj_bias: onnxscript.FLOAT,
    add_k_proj_weight: onnxscript.FLOAT,
    add_k_proj_bias: onnxscript.FLOAT,
    add_v_proj_weight: onnxscript.FLOAT,
    add_v_proj_bias: onnxscript.FLOAT,

    norm_added_q_weight: onnxscript.FLOAT,
    norm_added_q_eps: float,
    norm_added_q_elementwise_affine: bool,

    norm_added_k_weight: onnxscript.FLOAT,
    norm_added_k_eps: float,
    norm_added_k_elementwise_affine: bool,

    # Weights and Biases for Output Projections (`attn.to_out[0]`, `attn.to_out[1]`)
    to_out_0_weight: onnxscript.FLOAT,
    to_out_0_bias: onnxscript.FLOAT,
    to_out_1_dropout_p: float, # Dropout ratio for self.to_out[1]

    # Other flags
    attn_context_pre_only: bool, # From attn.context_pre_only
    attn_added_kv_proj_dim: int, # From attn.added_kv_proj_dim (to determine if add_q_proj etc. exist)
    to_add_out_weight: onnxscript.FLOAT, # For attn.to_add_out
    to_add_out_bias: onnxscript.FLOAT,
    ):
    residual = hidden_states
    batch_size = ops.Shape(hidden_states)[0]
    
    # Check if encoder_hidden_states is "None" (represented by empty tensor)
    # This conditional handling for ONNX `If` is tricky. I'll use Python `if` for tracing.
    # A true ONNX graph might trace both paths if this is a dynamic input.
    encoder_is_not_none = ops.Cast(ops.Size(encoder_hidden_states) > 0, to=9) # to BOOL

    # --- Sample Projections (Query, Key, Value from hidden_states) ---
    query = ops.Add(ops.MatMul(hidden_states, ops.Transpose(to_q_weight, perm=[1, 0])), to_q_bias)
    key = ops.Add(ops.MatMul(hidden_states, ops.Transpose(to_k_weight, perm=[1, 0])), to_k_bias)
    value = ops.Add(ops.MatMul(hidden_states, ops.Transpose(to_v_weight, perm=[1, 0])), to_v_bias)

    # Reshape for multi-head attention and transpose
    # (batch_size, seq_len, inner_dim) -> (batch_size, seq_len, heads, head_dim) -> (batch_size, heads, seq_len, head_dim)
    seq_len = ops.Shape(hidden_states)[1]
    
    query = ops.Transpose(ops.Reshape(query, ops.Concat([batch_size, seq_len, attn_heads, attn_head_dim], axis=0)), perm=[0, 2, 1, 3])
    key = ops.Transpose(ops.Reshape(key, ops.Concat([batch_size, seq_len, attn_heads, attn_head_dim], axis=0)), perm=[0, 2, 1, 3])
    value = ops.Transpose(ops.Reshape(value, ops.Concat([batch_size, seq_len, attn_heads, attn_head_dim], axis=0)), perm=[0, 2, 1, 3])

    query = RMSNormOnnx(query, norm_q_weight, norm_q_eps, norm_q_elementwise_affine)
    key = RMSNormOnnx(key, norm_k_weight, norm_k_eps, norm_k_elementwise_affine)
# --- Context Projections (from encoder_hidden_states) ---
    # This block is conditional on `encoder_hidden_states is not None`
    # We will compute both paths and use `ops.If` or a conditional switch later if truly dynamic.
    # For tracing, it will trace with a non-empty encoder_hidden_states if provided.
    
    # Placeholder for conditional output to ensure full graph is traced if encoder_is_not_none can be dynamic.
    encoder_hidden_states_query_proj_out = ops.Constant(value_float=0.0, output_dtype=1)
    encoder_hidden_states_key_proj_out = ops.Constant(value_float=0.0, output_dtype=1)
    encoder_hidden_states_value_proj_out = ops.Constant(value_float=0.0, output_dtype=1)
    
    if encoder_is_not_none: # `if encoder_hidden_states is not None` branch
        encoder_hidden_states_query_proj = ops.Add(ops.MatMul(encoder_hidden_states, ops.Transpose(add_q_proj_weight, perm=[1,0])), add_q_proj_bias)
        encoder_hidden_states_key_proj = ops.Add(ops.MatMul(encoder_hidden_states, ops.Transpose(add_k_proj_weight, perm=[1,0])), add_k_proj_bias)
        encoder_hidden_states_value_proj = ops.Add(ops.MatMul(encoder_hidden_states, ops.Transpose(add_v_proj_weight, perm=[1,0])), add_v_proj_bias)

        # Reshape and transpose for multi-head attention
        enc_seq_len = ops.Shape(encoder_hidden_states)[1]
        
        encoder_hidden_states_query_proj = ops.Transpose(ops.Reshape(encoder_hidden_states_query_proj, ops.Concat([batch_size, enc_seq_len, attn_heads, attn_head_dim], axis=0)), perm=[0, 2, 1, 3])
        encoder_hidden_states_key_proj = ops.Transpose(ops.Reshape(encoder_hidden_states_key_proj, ops.Concat([batch_size, enc_seq_len, attn_heads, attn_head_dim], axis=0)), perm=[0, 2, 1, 3])
        encoder_hidden_states_value_proj = ops.Transpose(ops.Reshape(encoder_hidden_states_value_proj, ops.Concat([batch_size, enc_seq_len, attn_heads, attn_head_dim], axis=0)), perm=[0, 2, 1, 3])

        # Apply RMSNorm if enabled (norm_added_q, norm_added_k)
        encoder_hidden_states_query_proj = RMSNorm(encoder_hidden_states_query_proj, norm_added_q_weight, norm_added_q_eps, norm_added_q_elementwise_affine)
        encoder_hidden_states_key_proj = RMSNorm(encoder_hidden_states_key_proj, norm_added_k_weight, norm_added_k_eps, norm_added_k_elementwise_affine)

        # Concatenate query, key, value from sample and context
        query = ops.Concat([query, encoder_hidden_states_query_proj], dim=2) # Concat along sequence length (dim=2)
        key = ops.Concat([key, encoder_hidden_states_key_proj], dim=2)
        value = ops.Concat([value, encoder_hidden_states_value_proj], dim=2)
        # --- Scaled Dot-Product Attention ---
    # `dropout_p=0.0, is_causal=False` are fixed.
    hidden_states_attn = ops.ScaledDotProductAttention(
        query,
        key,
        value,
        attention_mask,
        is_causal=False
    )
   
    # Reshape output back to (batch_size, seq_len, total_heads * head_dim)
    # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states_attn = ops.Transpose(hidden_states_attn, perm=[0, 2, 1, 3])
    hidden_states_attn = ops.Reshape(hidden_states_attn, ops.Concat([batch_size, -1, ops.Mul(attn_heads, attn_head_dim)], axis=0))
    final_hidden_states = ops.Constant(value_float=0.0, output_dtype=1)
    final_encoder_hidden_states = ops.Constant(value_float=0.0, output_dtype=1)

    if encoder_is_not_none: # If cross-attention was performed, split the output
        sample_output_len = ops.Shape(residual)[1] # Length of the original 'sample' sequence
        total_output_len = ops.Shape(hidden_states_attn)[1]
# Slice `hidden_states_attn` into two parts
        final_hidden_states = ops.Slice(
            hidden_states_attn,
            starts=ops.Constant(value_ints=[0, 0]),
            ends=ops.Constant(value_ints=[ops.Shape(hidden_states_attn)[0], sample_output_len]),
            axes=ops.Constant(value_ints=[0, 1]),
            steps=ops.Constant(value_ints=[1, 1])
        )
        final_encoder_hidden_states = ops.Slice(
            hidden_states_attn,
            starts=ops.Constant(value_ints=[0, sample_output_len]),
            ends=ops.Constant(value_ints=[ops.Shape(hidden_states_attn)[0], total_output_len]),
            axes=ops.Constant(value_ints=[0, 1]),
            steps=ops.Constant(value_ints=[1, 1])
        )
    else: # If no cross-attention, the attention output is just for `hidden_states`
        final_hidden_states = hidden_states_attn
        final_encoder_hidden_states = encoder_hidden_states # encoder_hidden_states remains what it was (e.g., empty tensor)

    # --- Post-Attention Processing ---
    # Apply attn.to_add_out if not context_pre_only and encoder_hidden_states was present
    if (not attn_context_pre_only) and encoder_is_not_none:
        final_encoder_hidden_states = ops.Add(ops.MatMul(final_encoder_hidden_states, ops.Transpose(to_add_out_weight, perm=[1,0])), to_add_out_bias)

    # Apply attn.to_out[0] (Linear proj)
    final_hidden_states = ops.Add(ops.MatMul(final_hidden_states, ops.Transpose(to_out_0_weight, perm=[1,0])), to_out_0_bias)

    # Apply attn.to_out[1] (Dropout)
    final_hidden_states = ops.Dropout(final_hidden_states, ratio=to_out_1_dropout_p)
 # Return based on whether encoder_hidden_states was provided in the input
    # The output signature must be consistent for ONNX `If` operators.
    # So both outputs are always returned.
    return final_hidden_states, final_encoder_hidden_states