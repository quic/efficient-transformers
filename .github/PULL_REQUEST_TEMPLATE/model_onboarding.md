## Summary
<!-- What changed and why? -->

## Model Onboarding Checklist
<!-- Only required when a maintainer adds a check:model-* label. Check an item when completed or when the line explicitly says N/A. -->

- [ ] Architecture classified
- [ ] Nearest QEff wrapper identified
- [ ] Wiring updated or N/A
- [ ] Cache/runtime helpers validated or N/A
- [ ] Tests updated or N/A
- [ ] Docs/examples/configs updated or N/A
- [ ] Validation commands listed

### LLM
<!-- Required only with check:model-llm. -->
- [ ] LLM: HF/QEff/ORT parity validated
- [ ] LLM: KV cache/prefill/decode validated
- [ ] LLM: continuous batching validated or N/A
- [ ] LLM: ONNX subfunctions validated or N/A

### MoE
<!-- Required only with check:model-moe. -->
- [ ] MoE: router/top-k validated
- [ ] MoE: expert packing validated
- [ ] MoE: decode and prefill validated
- [ ] MoE: subfunction/einsum export validated

### VLM
<!-- Required only with check:model-vlm. -->
- [ ] VLM: text-side parity validated
- [ ] VLM: full export validated
- [ ] VLM: single/dual QPC validated or N/A
- [ ] VLM: image preprocessing/token insertion validated

### Embedding
<!-- Required only with check:model-embedding. -->
- [ ] Embedding: tensor parity validated
- [ ] Embedding: pooling/attention-mask validated
- [ ] Embedding: multi-seq-len validated or N/A
- [ ] Embedding: ONNX transforms validated

### Diffusion
<!-- Required only with check:model-diffusion. -->
- [ ] Diffusion: component parity/MAD validated
- [ ] Diffusion: generation output validated
- [ ] Diffusion: ONNX/QPC artifacts validated
- [ ] Diffusion: pipeline modes validated or N/A

## Tests Run
<!-- Paste exact commands and any environment/hardware notes. -->
