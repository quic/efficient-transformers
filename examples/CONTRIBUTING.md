# Contributing Examples

This guide explains how to add new examples to the QEfficient repository.

## When to Add an Example

Add a new example if:
- The model requires special configuration not covered by existing examples
- You're demonstrating a new feature or optimization technique
- The model has unique requirements (dependencies, image sizes, etc.)

Don't add an example if:
- The model works with existing generic examples (just use those)
- The only difference is the model name, you can include the model name in validated model list and model class readme file. 

## Text Generation: One Canonical Entry Point

Text-only language models are served by a single flag-driven script,
[`text_generation/basic_inference.py`](text_generation/basic_inference.py). It
covers dense, continuous-batching, MoE, GGUF, blocked-attention, disaggregated,
and MDP workloads as flags over the same
`QEFFAutoModelForCausalLM.from_pretrained` / `.compile` / `.generate` calls.

**Do not add a new per-model `.py` file under `text_generation/`.** To support a
new model or recipe:
- If it runs on the existing three-step API, express it as flags on
  `basic_inference.py` (see `python text_generation/basic_inference.py --help`
  and `--help-advanced`) and add the model to the validated model list — no code
  change needed.
- If it needs new tunable behaviour, add the flag/config to `basic_inference.py`
  and its shared parser in [`_common/args.py`](_common/args.py) so every recipe
  stays on one entry point.
- Only genuinely bespoke, one-off surgery that does not fit the canonical API
  belongs under `text_generation/advanced/`, with a README noting its
  scratchpad nature (see the existing GLM4 / Kimi-K2 recipes).

A reviewer will ask any new text-generation script to justify why it cannot be a
flag or config on `basic_inference.py`.

## Directory Structure

Place your example in the appropriate domain:
- `text_generation/` - Text-only language models (flags on `basic_inference.py`; see above)
- `image_text_to_text/` - Vision-language models
- `embeddings/` - Embedding models
- `audio/` - Speech and audio models
- `peft/` - Fine-tuning and adapter examples
- `performance/` - Optimization techniques



## File Requirements

### 1. Python Script

> Note: this standalone-script template applies to the vision, embeddings,
> audio, PEFT, and performance domains, and to `text_generation/advanced/`
> one-offs. New text-generation recipes go through `basic_inference.py` flags
> instead (see "Text Generation: One Canonical Entry Point" above).

Your example script should:
- Include the copyright header
- Use argparse for command-line arguments
- Provide clear error messages
- Print results in a readable format

Basic template:
```python
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from transformers import AutoTokenizer
from QEfficient import QEFFAutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Description of what this example does")
    parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="Hello", help="Input prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=1)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name)
    
    qpc_path = model.compile(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
    )
    
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=[args.prompt],
    )
    
    print(f"Generated: {exec_info.generated_texts[0]}")

if __name__ == "__main__":
    main()
```

### 2. README.md

Each model-specific example needs a README explaining:
- What the model does
- Any special requirements
- How to run it
- Expected output

Template:
```markdown
# [Model Name]

## Overview
Brief description of the model and what makes it special.

## Requirements
```bash
# For single package
pip install package-name==1.2.3

# For multiple packages
pip install package-name==1.2.3 another-package==4.5.6

# Or use a requirements.txt file
pip install -r requirements.txt
```

**Note:** Always specify exact versions to ensure reproducibility. Use `pip show package-name` to check installed versions.

## Usage
```bash
python inference.py --model-name [model-id] --prompt "Your prompt"
```

## Special Notes
Any model-specific considerations, limitations, or configuration details.

## References
- Model card: [link]
- Paper: [link] (optional)

## Code Guidelines

- Use clear variable names
- Add comments for non-obvious code
- Handle errors gracefully
- Follow existing code style in the repository
- Test your example before submitting

## Testing Your Example

Before submitting:
1. Run the example with default parameters
2. Test with different model sizes if applicable
3. Verify the README instructions work
4. Check that all dependencies are documented

## Submitting Your Contribution

Follow these steps to submit your example to the QEfficient repository:

### 1. Fork and Clone the Repository

First, fork the repository to your GitHub account, then clone your fork:

```bash
# Fork the repository on GitHub (click the "Fork" button)
# Then clone your fork
git clone git@github.com:YOUR_USERNAME/efficient-transformers.git
cd efficient-transformers

# Add upstream remote to keep your fork in sync
git remote add upstream git@github.com:quic/efficient-transformers.git
```

### 2. Create a Feature Branch

Create a descriptive branch for your changes:

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b add-[model-name]-example
```

### 3. Make Your Changes

Add your example files following the guidelines above:
- Python script with proper copyright header
- README.md with clear documentation
- requirements.txt (if needed)

### 4. Run Pre-commit Checks

Before committing, ensure your code passes all quality checks:

```bash
# Install pre-commit if not already installed
pip install pre-commit

# Run pre-commit on your changed files
pre-commit run --files path/to/your/file1.py path/to/your/file2.md
```

**Important:** If pre-commit reports any failures:
- Some issues will be auto-fixed (formatting, trailing whitespace, etc.)
- For issues that aren't auto-fixed, manually correct them
- Re-run `pre-commit run --files <files>` until all checks pass

### 5. Commit with Sign-off (DCO)

All commits must be signed off to comply with the Developer Certificate of Origin (DCO):

```bash
# Stage your changes
git add examples/your_domain/your_example.py
git add examples/your_domain/README.md

# Commit with sign-off
git commit -s --author "Your Name <your.email@example.com>" -m "Add [model-name] example

- Implements inference for [model-name]
- Includes documentation and usage examples
- Tested with [specific configurations]"
```

**Commit Message Guidelines:**
- Use a clear, descriptive title 
- Add a blank line, then detailed description if needed
- Always include the `-s` flag for DCO sign-off

### 6. Push to Your Fork

Push your branch to your forked repository:

```bash
git push origin add-[model-name]-example
```

### 7. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request" for your branch
3. Fill out the PR template with:
   - **Title:** Clear, descriptive title (e.g., "Add Llama-3.2-Vision example")
   - **Description:** 
     - What the example demonstrates
     - Why it's needed (what makes it different from existing examples)
     - Any special testing considerations
     - Link to model card or documentation
   - **Testing:** Describe how you tested the example

### 8. Ensure CI Checks Pass

After creating the PR, verify that all automated checks pass:

- ✅ **DCO Check:** Ensures all commits are signed off
- ✅ **Lint Check:** Code style and formatting validation
- ✅ **Tests:** Automated test suite (if applicable)

If any checks fail:
1. Review the error messages in the PR
2. Make necessary fixes in your local branch
3. Commit and push the fixes (with sign-off)
4. The PR will automatically update and re-run checks

### 9. Address Review Feedback

Maintainers will review your PR and may request changes:
- Make requested changes in your local branch
- Commit with sign-off and push to update the PR
- Respond to comments to facilitate discussion

## Questions

For questions or issues, open a GitHub issue or discussion.
