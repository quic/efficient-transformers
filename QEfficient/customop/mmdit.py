
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
def GELUOnnx(
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

    # Parameters derived from FeedForward __init__
    dim: int,           # `dim` from FeedForward init (input dimension to the block)
    dim_out: int,       # `dim_out` from FeedForward init (output dimension of the block)
    mult: int,          # `mult` from FeedForward init (used to calculate inner_dim)
    dropout_ratio: float, # `dropout` for nn.Dropout
    final_dropout: bool, # `final_dropout` bool
    # We no longer need 'activation_fn_type' as we're fixed to GELU-approximate

    # Weights for the internal components
    # Weights for the 'act_fn' (which is GELU in this case)
    act_fn_proj_weight: onnxscript.FLOAT, # This is the weight for GELU's internal Linear layer
    act_fn_proj_bias: onnxscript.FLOAT,   # This is the bias for GELU's internal Linear layer

    # Weights for the final Linear layer (self.net.append(nn.Linear(...)))
    project_out_weight: onnxscript.FLOAT,
    project_out_bias: onnxscript.FLOAT,
):
    # Calculate inner_dim as in PyTorch FeedForward.__init__
    # inner_dim = int(dim * mult)
    inner_dim_val = ops.Cast(ops.Mul(dim, mult), to=6) # 6 is ONNX INT64
    # 1. Apply act_fn (which is GELUOnnx here)
    # The GELUOnnx function handles its own internal projection.
    # The output dimension of GELU's internal projection is `inner_dim`.
    ff_output = GELU(
        hidden_states,
        act_fn_proj_weight,
        act_fn_proj_bias,
    )

    # 2. Apply first Dropout
    # For inference, dropout_ratio is typically 0.0, which means ops.Dropout acts as ops.Identity.
    # If opset is < 12 and dropout_ratio is 0.0, it might be removed by optimizers.
    ff_output = ops.Dropout(ff_output, ratio=dropout_ratio)

    # 3. Apply project out (final Linear layer)
    # The input to this linear layer is `ff_output`, which has shape [..., inner_dim_val].
    # The output dimension of this linear layer is `dim_out`.
    ff_output = ops.MatMul(ff_output, ops.Transpose(project_out_weight, perm=[1, 0]))
    ff_output = ops.Add(ff_output, project_out_bias)

    # 4. Apply final Dropout (if final_dropout is True)
    if final_dropout:
        ff_output = ops.Dropout(ff_output, ratio=dropout_ratio) # Re-use dropout_ratio

    return ff_output




