
import onnxscript
import torch
import torch.nn as nn

# Define your ONNXScript Opset domain
# All custom ONNX functions (like norms, attention, FF) should ideally be in the same domain.
CUSTOM_OPSET = onnxscript.values.Opset(domain="com.qualcomm.cloud", version=1)
# Import the ONNX Script opset for version 13
ops = getattr(onnxscript, "opset" + str(13))


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def SD35AdaLayerNormZeroX(
    hidden_states: onnxscript.FLOAT,
    emb: onnxscript.FLOAT, # temb in the JointTransformerBlock forward
    linear_weight: onnxscript.FLOAT,
    linear_bias: onnxscript.FLOAT,
    norm_epsilon: float, # For LayerNorm's epsilon (elementwise_affine=False means no weight/bias on norm)
):
    # This is `self.silu = nn.SiLU(); self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias)`
    # then chunk, then LayerNorm, then operations with chunked outputs.

    # 1. emb = self.linear(self.silu(emb))
    silu_emb = ops.Mul(emb, ops.Sigmoid(emb)) # Equivalent to nn.SiLU()
    linear_out = ops.MatMul(silu_emb, ops.Transpose(linear_weight, perm=[1, 0])) # PyTorch Linear behavior (input@W.T)
    linear_out = ops.Add(linear_out, linear_bias)

    # 2. Chunk `linear_out` into 9
    # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2
    # Determine chunk size dynamically, assuming equal chunks.
    output_dim_linear = ops.Shape(linear_out)[-1]
    chunk_size = ops.Cast(output_dim_linear / 9, to=6) # Cast to Int64

    # The ops.Split operator requires explicit sizes for each chunk
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
    # self.norm is nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
    # ONNX opset16 LayerNormalization has elementwise_affine=True by default.
    # To simulate elementwise_affine=False, pass `None` for scale/bias, or `ops.Constant(value_float=1.0)`/`ops.Constant(value_float=0.0)`
    # The LayerNormalization operator requires scale and bias inputs, even if they are identity.
    # Let's assume the scale is implicit 1.0 and bias implicit 0.0 for elementwise_affine=False.
    # The easiest way is to apply LayerNormalization without learnable weights/bias:
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


 # Use the appropriate opset



@onnxscript.script(CUSTOM_OPSET)
def AdaLayerNormZero(
    x: onnxscript.FLOAT, # Corresponds to 'hidden_states' in your PyTorch code
    emb: onnxscript.FLOAT, # This is the 'emb' after potential `self.emb` processing
                           # (i.e., the input to self.linear)

    linear_weight: onnxscript.FLOAT, # Weight for self.linear
    linear_bias: onnxscript.FLOAT,   # Bias for self.linear

    norm_epsilon: float,             # eps for self.norm (LayerNorm)
    # If your LayerNorm can be `fp32_layer_norm` (as per `norm_type` in init),
    # its ONNX equivalent would be a specialized operator, or `ops.LayerNormalization`
    # with the appropriate settings/conversions. For now, assuming `layer_norm`.
):
    # This ONNXScript function assumes `emb` is already processed by `self.emb` (if it exists)
    # and is the direct input to `self.linear`.

    # 1. `emb = self.linear(self.silu(emb))`
    silu_emb = ops.Mul(emb, ops.Sigmoid(emb)) # Equivalent to nn.SiLU()

    # Apply the linear layer
    # PyTorch's nn.Linear internally does `input @ weight.T + bias`
    # So, we need to transpose the weight for ops.MatMul(input, weight)
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
    # `self.norm` is `nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)`
    # ONNX LayerNormalization operator requires `scale` and `bias` inputs.
    # For `elementwise_affine=False`, these should be identity tensors.
    norm_x = ops.LayerNormalization(
        x,
        scale=ops.Constant(value_float=1.0, output_dtype=1), # float type (ONNX FLOAT)
        bias=ops.Constant(value_float=0.0, output_dtype=1),  # float type (ONNX FLOAT)
        epsilon=norm_epsilon
    )

    # Apply the scaling and shifting: `norm_x * (1 + scale_msa[:, None]) + shift_msa[:, None]`
    # Use ops.Unsqueeze for `[:, None]` equivalent
    scaled_shifted_x = ops.Add(
        ops.Mul(norm_x, ops.Add(ops.Constant(value_float=1.0), ops.Unsqueeze(scale_msa, axes=[1]))),
        ops.Unsqueeze(shift_msa, axes=[1])
    )

    # Return signature: x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    return scaled_shifted_x, gate_msa, shift_mlp, scale_mlp, gate_mlp







