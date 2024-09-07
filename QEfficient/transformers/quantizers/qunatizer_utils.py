from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING
from transformers.integrations.awq import AWQ_SCALES_MAPPINGS
from torch import nn
import copy
import torch

def replace_transformers_quantizers():
    from  QEfficient.transformers.quantizers.quantizer_awq import QEffAwqQuantizer, QEffAwqConfig
    from  QEfficient.transformers.quantizers.quantizer_gptq import QEffGPTQQuantizer, QEffGPTQConfig 

    AUTO_QUANTIZER_MAPPING.update({"gptq": QEffGPTQQuantizer})
    AUTO_QUANTIZATION_CONFIG_MAPPING.update({"gptq": QEffGPTQConfig})
    AUTO_QUANTIZER_MAPPING.update({"awq": QEffAwqQuantizer})
    AUTO_QUANTIZATION_CONFIG_MAPPING.update({"awq": QEffAwqConfig})
    
    
def get_keys_to_not_convert(model):
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = copy.deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # If there is not tied weights, we want to keep the lm_head（output_embedding) in full precision
    if not has_tied_params:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module

    # otherwise, no tied weights, no output embedding defined, simply keep the last module in full precision
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names

def find_tied_parameters(model: nn.Module, **kwargs):
    # Initialize result and named_parameters before recursing.
    named_parameters = kwargs.get("named_parameters", None)
    prefix = kwargs.get("prefix", "")
    result = kwargs.get("result", {})

    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        # A tied parameter will not be in the full `named_parameters` seen above but will be in the `named_parameters`
        # of the submodule it belongs to. So while recursing we track the names that are not in the initial
        # `named_parameters`.
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        if new_name not in result:
                            result[new_name] = []
                        result[new_name].append(full_name)

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        find_tied_parameters(child, named_parameters=named_parameters, prefix=child_name, result=result)

    return [sorted([weight] + list(set(tied))) for weight, tied in result.items()]


def replace_linear_layer_with_target_layer(
    model: torch.nn.Module,
    target_cls,
    quantization_config=None,
    modules_to_not_convert=None,
    current_key_name=None,
    has_been_replaced=False,
    
):
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    # target_cls = WQLinear_GEMM

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = target_cls(
                    bits=quantization_config.bits,
                    groupsize=quantization_config.group_size,
                    infeatures=in_features,
                    outfeatures=out_features,
                    bias=module.bias is not None,
                    # dev=module.weight.device,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_linear_layer_with_target_layer(
                module,
                target_cls,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


def replace_quantization_scales(model, model_type):
    if model_type not in AWQ_SCALES_MAPPINGS:
        return model
    for name, module in model.named_children():
        act_name = AWQ_SCALES_MAPPINGS[model_type]["act"]
        layer_before_act_name = AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"]
        if name == act_name and hasattr(model, layer_before_act_name):
            layer_before_act = getattr(model, AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"])
            size = layer_before_act.out_features
            scale_like = torch.ones(size)
            model._modules[name] = ScaledActivation(module, scale_like)
            replace_quantization_scales(module, model_type)
    return model


def dequantize_gemm1(qweight, qzeros, scales, bits, group_size , quant, g_idx):
    # Unpack the qweight and qzeros tensors
    iweight, scales, izeros = unpack_weights(qweight, qzeros, scales, bits, group_size,quant,g_idx)

    # fp16 weights
    if quant == "awq":
        scales = scales.repeat_interleave(group_size, dim=0)
        izeros = izeros.repeat_interleave(group_size, dim=0)
        iweight = (iweight - izeros) * scales

    return iweight


def unpack_weights1(qweight, qzeros, scales, bits, group_size, quant: str,g_idx):
 
    iweights, izeros = unpack_weights_and_zeros1(qweight, qzeros, bits,quant)
    # Reverse the order of the iweight and izeros tensors
    # overflow checks
    
    print(iweights.shape)
    # import ipdb; ipdb.set_trace()
    if quant == "gptq":
        izeros = torch.bitwise_and(izeros, (2 ** bits) - 1).view(-1, 1, izeros.size(1) * izeros.size(2))
        izeros = izeros.view(izeros.shape[0], -1)
        
        iweights=iweights.permute(0,2,1)
        
        iweights=iweights.reshape(-1,iweights.shape[2])
    elif quant == "awq":
        izeros = izeros.view(izeros.shape[0], -1)
        izeros = torch.bitwise_and(izeros, (2**bits) - 1)
        
        iweights = iweights.view(-1,iweights.shape[1])
        
    
    # if quant =="gptq":
    #     iweights=iweights[0].permute(0,2,1)
    #     iweights=iweights.reshape(iweights.shape[2],iweights.shape[2])
    # elif quant =="awq":
    #     iweights = iweights.view(iweights.shape[0], -1)
    
    iweights = torch.bitwise_and(iweights, (2 ** bits) - 1)

    # if quant=="gptq":
    #     scales = scales.view(-1, 1, scales.size(-1))
    #     scales=scales.view(scales.shape[0],-1)
    #     scale_zeros = izeros * scales
    #     scale_mat = scales[g_idx]
    #     scale_zeros_mat = scale_zeros[g_idx]
    #     qdq_weight_T = iweights * scale_mat - scale_zeros_mat.float()
    #     qdq_weight_T=torch.transpose(qdq_weight_T,0,1)
    
    return iweights,scales, izeros


def unpack_weights_and_zeros1(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int, quant: str):
    if quant=='gptq':
        shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device)
        # shifts = torch.arange(0, 32, bits)
    else:
        shifts = torch.arange(0, 32, bits)
    # unpacking columnwise
    if quant == 'gptq':
        iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int32)  # smallest dtype available
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int32  # smallest dtype available
        )
    elif quant == 'awq':
        iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )  
    return iweights, izeros


def reverse_awq_order(int_weights: torch.Tensor, int_zeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        int_weights.shape[-1],
        dtype=torch.int32,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    int_zeros = int_zeros[:, reverse_order_tensor]
    int_weights = int_weights[:, reverse_order_tensor]

    return int_weights, int_zeros

def unpack_weights_and_zeros(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int, quant: str):
    shifts = torch.arange(0, 32, bits)

    # unpacking weights column-wise
    int_weights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    int_weights = int_weights.reshape(int_weights.shape[0], -1)

    # unpacking zeros column-wise
    int_zeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    int_zeros = int_zeros.reshape(int_zeros.shape[0], -1)
    
    if quant == 'awq': 
        return reverse_awq_order(int_weights,int_zeros,bits)
    
    return int_weights, int_zeros



def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    scales, int_weight, int_zeros = unpack_weights(qweight, qzeros, scales, bits, 'awq')

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    int_zeros = int_zeros.repeat_interleave(group_size, dim=0)

    int_weight = (int_weight - int_zeros) * scales

    return int_weight

def dequantize_gptq(qweight, qzeros, scales, bits, g_idx):
    
    scales, int_weight, int_zeros = unpack_weights(qweight, qzeros, scales, bits, 'gptq')
    
    scales = scales.view(-1, 1, scales.size(-1))
    scales=scales.view(scales.shape[0],-1)
    scale_zeros = int_zeros * scales
    scale_mat = scales[g_idx]
    scale_zeros_mat = scale_zeros[g_idx]
    int_weight = int_weight.T * scale_mat - scale_zeros_mat.float()
    # int_weight = torch.transpose(int_weight,0,1)
    return int_weight, scales, int_zeros
    

def unpack_weights(qweight, qzeros, scales, bits, quant):
    # import ipdb; ipdb.set_trace()
    int_weight, int_zeros = unpack_weights_and_zeros(qweight, qzeros, bits, quant)

    # overflow checks
    int_weight = torch.bitwise_and(int_weight, (2**bits) - 1)
    int_zeros = torch.bitwise_and(int_zeros, (2**bits) - 1)

    return scales, int_weight, int_zeros
