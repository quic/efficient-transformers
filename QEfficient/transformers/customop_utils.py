custom_opset = onnxscript.values.Opset("com.qti.aisw.onnx", 1)
ops = onnxscript.opset13

@onnxscript.script(custom_opset)
def CustomRMSNorm(hidden_states: onnxscript.FLOAT, weight: onnxscript.FLOAT, epsilon: float):
    variance = ops.ReduceMean(ops.Pow(hidden_states, 2), axes=[-1], keepdims=1)
    epsilon = ops.Expand(epsilon, ops.Shape(variance))
    hidden_states = hidden_states * ops.Reciprocal(ops.Sqrt(variance + epsilon))
    return weight * hidden_states


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.onnx._internal.jit_utils.GraphContext,
        hidden_states: torch.Value,
        weight: torch.Value,
        epsilon: torch.Value,
    ) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


