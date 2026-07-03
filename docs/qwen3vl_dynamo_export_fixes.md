# Qwen3-VL Dynamo Export — Bug Fixes

All errors were encountered when exporting `QEffQwen3VLEncoderWrapper` and
`QEffQwen3VLDecoderWrapper` with `use_dynamo=True, use_onnx_subfunctions=True,
kv_offload=True`.

---

## Fix 1 — `deepstack_features` in `vision_dynamic_shapes`

**File:** `QEfficient/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
`get_onnx_dynamic_shapes()`

### Error
```
torch._dynamo.exc.UserError: When `dynamic_shapes` is specified as a dict,
its top-level keys must be the arg names ['pixel_values', 'image_grid_thw']
of `inputs`, but here they are ['pixel_values', 'image_grid_thw', 'deepstack_features'].
```

### Why
`torch.export` validates that `dynamic_shapes` keys exactly match the **input**
argument names of the function being exported.
`deepstack_features` is an **output** of `QEffQwen3VLEncoderWrapper.forward`
(which only takes `pixel_values` and `image_grid_thw`). It was incorrectly
included in `vision_dynamic_shapes` by copy from `vision_dynamic_axes`.

The TorchScript path (`use_dynamo=False`) accepts `dynamic_axes` entries for
**both inputs and outputs** (per PyTorch docs: "KEY: an input or output name"),
so it was valid there. The dynamo `dynamic_shapes` parameter is inputs-only.

### Fix
Remove `deepstack_features` from `vision_dynamic_shapes` in the `kv_offload=True`
branch. It correctly remains in `lang_dynamic_shapes` because it is an input to
`QEffQwen3VLDecoderWrapper.forward`.

---

## Fix 2 — `GuardOnDataDependentSymNode` from `int(...item())`

**File:** `modeling_qwen3_vl.py` — `QEffQwen3VLVisionModel.rot_pos_emb()` line 155

### Error
```
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
Could not extract specialized integer from data-dependent expression u0.
Caused by: rot_pos_emb:155 in forward, code:
  total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
```

### Why
`grid_thw` was re-assigned to `torch.tensor(grid_thw.shape)` — a tensor built
from symbolic dims. `torch.prod(...).sum().item()` tried to extract a Python int
from a symbolic tensor during `torch.export` tracing, which is not allowed.

### Fix
These three lines were dead code — `pos_ids` was unconditionally overwritten at
line 177. Remove the `grid_thw` reassignment and the `total_tokens`/`pos_ids`
allocation entirely.

---

## Fix 3 — `batch_size` specialised to 1 via `max(grid_thw.shape)` and `if num_frames > 1`

**File:** `modeling_qwen3_vl.py` — `rot_pos_emb()` lines 149 and 175

### Error
```
torch._dynamo.exc.UserError: Constraints violated (batch_size, grid_h, grid_w, time)!
  - Not all values of batch_size in the specified range batch_size <= 1024
    satisfy the generated guard L['image_grid_thw'].size()[0] <= 2
  - solving the guards for batch_size resulted in a specialized value of 1.
Suggested fixes:
  time = Dim('time', max=1)
  batch_size = 1
  grid_h = 22
  grid_w = 34
```

### Why
Two problems in `rot_pos_emb`:

1. `max(grid_thw.shape)` — Python's `max()` compares all four symbolic dims
   `(bs, t, h, w)` pairwise. With example `(2, 1, 22, 34)`, this generates
   comparison guards such as `bs <= w` and `bs <= h`, which — combined —
   resolve `batch_size` to a concrete value.

2. `if num_frames > 1:` — evaluating a Python boolean from the symbolic `t`
   dim forces `time` to be concrete (= 1 from the dummy input), which cascades
   to specialise `batch_size` too.

### Fix
- Replace `max(grid_thw.shape)` with the static config value
  `self.num_grid_per_side` (no symbolic comparison needed).
- Replace `if num_frames > 1: coords = coords.repeat(num_frames, 1)` with an
  unconditional `coords = coords.repeat(num_frames, 1)` so no Python boolean
  guard is generated on `num_frames`.

---

## Fix 4 — `batch_size` specialised to 1 via `.squeeze().expand(bs)`

**File:** `modeling_qwen3_vl.py` — `QEffQwen3VLVisionModel.forward()` lines 257–259

### Error
```
torch._dynamo.exc.UserError: Constraints violated (batch_size)!
  - Not all values of batch_size in the specified range batch_size <= 1024
    satisfy the generated guard L['image_grid_thw'].size()[0] <= 2
  - solving the guards for batch_size resulted in a specialized value of 1.
```

### Why
```python
t = torch.arange(t, t + 1).squeeze().expand(bs)
h = torch.arange(h, h + 1).squeeze().expand(bs)
w = torch.arange(w, w + 1).squeeze().expand(bs)
```
`torch.arange(x, x+1)` creates a 1-element tensor; `.squeeze()` makes it a
0-d scalar tensor. Calling `.expand(bs)` on a 0-d tensor is only valid when
`bs == 1` — dynamo generates the guard `batch_size <= 1`, conflicting with the
declared `Dim("batch_size", min=1, max=1024)`.

### Fix
Replace with `torch.full((bs,), value, dtype=torch.int64)` which directly
creates a 1-D tensor of length `bs` without any guard on `bs`.

```python
t = torch.full((bs,), t, dtype=torch.int64)
h = torch.full((bs,), h, dtype=torch.int64)
w = torch.full((bs,), w, dtype=torch.int64)
```

---

## Fix 5 — `pixel_values` dummy shape not scaled by `batch_size`

**File:** `modeling_qwen3_vl.py` — `get_dummy_inputs()` line 901

### Error
```
torch._dynamo.exc.UserError: Constraints violated (batch_size)!
  - solving the guards for batch_size resulted in a specialized value of 1.
```

### Why
`inputs_shapes["image_grid_thw"] = (bs, 1, 22, 34)` was correctly using `bs`,
but `inputs_shapes["pixel_values"] = (748, 1536)` was hardcoded for a single
image. `748 = grid_h × grid_w = 22 × 34` is the number of patches per image.
For `bs` images the encoder receives `748 * bs` patch rows.

When `torch.export` traces with `bs=2` but `pixel_values.shape[0]=748`,
`split_size = 748 / bs` only yields an integer when `bs=1`, so dynamo generates
the guard `batch_size == 1`.

### Fix
```python
inputs_shapes["pixel_values"] = (748 * bs, 1536)
```

---

## Fix 6 — `rot_pos_emb` generating only single-image token count

**File:** `modeling_qwen3_vl.py` — `rot_pos_emb()` lines 175–178

### Error
```
RuntimeError: The size of tensor a (64) must match the size of tensor b
(s26: hint=32) at non-singleton dimension 2
```
in `apply_rotary_pos_emb_vision`: `q * cos` shape mismatch.

### Why
After fix 5, `patch_embed` produces `748 * bs` token embeddings (one row per
patch across all batch images). But `rot_pos_emb` still only generated `748`
positional embeddings (one image worth). When `forward` does
`rotary_pos_emb.reshape(seq_len, -1)` with `seq_len = 748 * bs`, the reshape
halved the embedding dim from 64 to 32.

### Fix
After the existing `num_frames` repeat, add a `bs` repeat so all batch images
get positional embeddings:
```python
coords = coords.repeat(num_frames, 1)
coords = coords.repeat(bs, 1)  # cover all batch images
```

---

## Fix 7 — FakeTensor `_tensor_constant0` from boolean-indexed assignment

**File:** `modeling_qwen3_vl.py` — `QEffQwen3VLVisionAttention.forward()` lines 335–336

### Error
```
TypeError: Cannot take content out from the FakeTensor ('_tensor_constant0').
onnx_ir.serde.SerdeError: Error calling serialize_function_into
```

### Why
```python
final_mask = torch.ones((seq_len, seq_len), dtype=self.config.dtype)
final_mask[block_mask.any(dim=0)] = 0
```
Boolean-indexed assignment `tensor[mask] = 0` lowers to
`aten::index_put(tensor, [mask], scalar_0)`. The scalar `0` is wrapped as a 0-d
tensor constant node (`_tensor_constant0`). Inside a `@nested_compile_region`
subfunction this node is a FakeTensor with no real data — ONNX serialisation
calls `.tobytes()` on it and fails.

### Fix
Replace the three-line mask construction with a single `torch.where` that
embeds the scalar as an inline op attribute (not a separate tensor node):
```python
valid = block_mask.any(dim=0)
attention_mask = torch.where(
    valid,
    torch.zeros((seq_len, seq_len), dtype=q.dtype, device=q.device),
    torch.finfo(q.dtype).min
).unsqueeze(0)
```

---

## Fix 8 — Compiler error from `flatten(0, 4)` baking static shape 748

**File:** `modeling_qwen3_vl.py` — `fast_pos_embed_interpolate()` line 235

### Error
```
QAIC_ERROR: [Operator-'node__unsafe_view']:
Reshape: input shape (1, 11, 17, 2, 2, 748) and output shape (748, 1024)
have different number of elements (in 559504 vs. out 765952)
```

### Why
`.flatten(0, 4)` on the permuted tensor lowered to `_unsafe_view([748, 1024])`
with the first dim baked as the concrete value `748` from the dummy input
(`t * h//m * m * w//m * m = 1 * 11 * 2 * 17 * 2 = 748`). When the compiler
processes the downstream `expand(bs, -1, -1).reshape(-1, C)` path with `bs > 1`,
it tried to reshape the static 6-D permuted tensor
`(1, 11, 17, 2, 2, 1024)` into `[748, 1024]` — wrong because the total elements
are `559504 ≠ 765952`.

### Fix
Replace `.flatten(0, 4)` with `.contiguous().view(-1, embed_dim)`:
```python
embed_dim = pos_embed.shape[-1]
pos_embed = (
    pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, embed_dim)
    .permute(0, 1, 3, 2, 4, 5)
    .contiguous()
    .view(-1, embed_dim)
)
```

---

## Fix 9 — `split([h * w])` baking static `748` in `split_with_sizes`

**File:** `modeling_qwen3_vl.py` — `fast_pos_embed_interpolate()` line 225

### Error
```
QAIC_ERROR: [Operator-'node_view_10']:
Reshape: input shape (748, 748) and output shape (1, 11, 2, 17, 2, 1024)
have different number of elements (in 559504 vs. out 765952)
```

### Why
`patch_pos_embeds.split([h * w])` — constructing a Python list `[h * w]` from
symbolic ints `h` and `w` forces their concrete evaluation at trace time,
producing `split_with_sizes(..., [748])` with the static value baked in. The
downstream `.view(t, h//m, ...)` then also used the static-valued intermediate,
causing the compiler to see wrong shapes for `bs > 1`.

### Fix
Replace `split([h * w])` with `narrow(0, 0, h * w)`, which takes a symbolic
integer directly as its length argument keeping it as a symbolic expression:
```python
patch_pos_embeds = patch_pos_embeds.narrow(0, 0, h * w)
```

---

## Fix 10 — `select_scatter` from `attention_mask[0] = final_mask`

**File:** `modeling_qwen3_vl.py` — `QEffQwen3VLVisionAttention.forward()` line 348

### Error
```
QAIC_ERROR: [Operator-'node_select_scatter']: ...
Unable to AddNodesToGraphFromModel
```

### Why
`attention_mask[0] = final_mask` — indexed assignment on a pre-allocated tensor
lowers to `aten::select_scatter(full, copy, dim=0, index=0)`. This is an
unsupported op in the QAIC compiler inside a subfunction.

### Fix
Eliminate the pre-allocation and in-place write entirely. Build `attention_mask`
directly:
```python
attention_mask = torch.where(
    valid,
    torch.zeros((seq_len, seq_len), dtype=q.dtype, device=q.device),
    torch.finfo(q.dtype).min
).unsqueeze(0)
```
This was combined with fix 7 into a single refactor.

---

## Fix 11 — `_local_scalar_dense` non-constant shape in Reshape

**File:** `modeling_qwen3_vl.py` — `QEffQwen3VLEncoderWrapper.forward()` line 684

### Error
```
QAIC_ERROR: [Operator-'node_view_27']:
Reshape: Non-constant shape tensor not supported.
```

### Why
```python
split_size = torch.floor_divide(torch.tensor(image_embeds.size(0)), bs)
```
`torch.tensor(image_embeds.size(0))` wraps a symbolic int into a 0-d tensor.
`torch.floor_divide` on two 0-d tensors produces a 0-d tensor. When that tensor
is used as a shape argument in `.reshape(bs, split_size, C)`, dynamo emits
`_local_scalar_dense` to extract the scalar — which the ONNX graph exposes as a
dynamic shape tensor. QAIC requires Reshape shape inputs to be constant.

### Fix
Compute `split_size` as a pure symbolic integer division, never going through a
tensor:
```python
split_size = image_embeds.size(0) // bs
```
Both operands are symbolic ints; `//` stays in the symbolic integer domain and
the ONNX exporter represents it inline as a constant-foldable shape expression.

---

## Summary table

| # | Error type | Source line | Root cause | Fix |
|---|---|---|---|---|
| 1 | `UserError` dynamic_shapes key mismatch | `get_onnx_dynamic_shapes` | `deepstack_features` is encoder output, not input | Remove from `vision_dynamic_shapes` |
| 2 | `GuardOnDataDependentSymNode` | `rot_pos_emb:155` | `.item()` on symbolic tensor | Remove dead code (total_tokens / pos_ids) |
| 3 | `batch_size` specialised to 1 | `rot_pos_emb:149,175` | `max(grid_thw.shape)` + `if num_frames > 1` | Use `self.num_grid_per_side`; unconditional `.repeat` |
| 4 | `batch_size` specialised to 1 | `forward:257-259` | `.squeeze().expand(bs)` on 0-d tensor | Use `torch.full((bs,), value)` |
| 5 | `batch_size` specialised to 1 | `get_dummy_inputs:901` | `pixel_values` shape not scaled by `bs` | `(748 * bs, 1536)` |
| 6 | RoPE shape mismatch at multiplication | `rot_pos_emb` | `coords` not repeated for all `bs` images | `coords = coords.repeat(bs, 1)` |
| 7 | `SerdeError` FakeTensor `_tensor_constant0` | `VisionAttention:335-336` | `tensor[mask] = 0` → `index_put` with 0-d tensor const | Rewrite with `torch.where` |
| 8 | Compiler Reshape element count mismatch | `fast_pos_embed:235` | `.flatten(0,4)` bakes static 748 | `.contiguous().view(-1, embed_dim)` |
| 9 | Compiler Reshape element count mismatch | `fast_pos_embed:225` | `split([h*w])` bakes static 748 | `narrow(0, 0, h*w)` |
| 10 | Compiler `select_scatter` unsupported | `VisionAttention:348` | `tensor[0] = value` → `select_scatter` | Direct `torch.where(...).unsqueeze(0)` |
| 11 | Compiler non-constant Reshape shape | `EncoderWrapper:684` | `torch.tensor(size) // bs` → `_local_scalar_dense` | `image_embeds.size(0) // bs` |
