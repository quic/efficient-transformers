## Contributing to PROJECT

Hi there!
We're thrilled that you'd like to contribute to this project.
Your help is essential for keeping this project great and for making it better.


## Submitting Your Contribution

Follow these steps to submit your example to the QEfficient repository:

1. Please read our [code of conduct](CODE-OF-CONDUCT.md) and [license](LICENSE).

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
git checkout -b <branch-name>
```

### 3. Make Your Changes

When making changes to the codebase:

- **Follow Existing Design Patterns**
  - Review similar implementations before creating new code
  - Maintain consistency with the project's architecture and coding style
  - Reuse existing utilities and base classes where applicable

- **Onboarding New Models**
  - For adding new model support, refer to the comprehensive guide: `examples/onboarding_guide/causallm/`
  - Follow the step-by-step process with code examples provided

- **Testing is Mandatory**
  - Add tests for all new features in the appropriate `tests/` subdirectory
  - Run tests locally before pushing: `pytest tests/path/to/your/test.py -v`
  - For model additions, verify all 4 pipeline stages (PyTorch HF → KV → ORT → AI 100) and make sure tokens are matching with refernce PyTorch HF 

- **Documentation**
  - **For New Features/Flags:**
    - Document usage in `docs/source/<appropriate-page>` with feature description and usage examples
    - Ensure documentation is clear enough for others to understand and use the feature
  - **For New Models:**
    - Test with basic inference scripts in the `examples/` folder
    - If specific changes are needed, create a dedicated example file
    - Update `docs/source/validate.md` with the model's HuggingFace card name and relevant details
  

- **Code Quality Checks**
  - Pre-commit hooks, DCO sign-off, and CI checks are covered in the following steps
  - Ensure you complete steps 4-8 before finalizing your PR

### 4. Run Pre-commit Checks

Before committing, ensure your code passes all quality checks:

```bash
# Install pre-commit and ruff if not already installed
pip install pre-commit
pip install ruff

# Run pre-commit on your changed files
pre-commit run --files path/to/your/file1.py path/to/your/file2.py 

# Run Ruff check 
ruff check
```

**Important:** If pre-commit reports any failures:
- Some issues will be auto-fixed (formatting, trailing whitespace, etc.)
- For issues that aren't auto-fixed, manually correct them
- Re-run `pre-commit run --files <files>` or `ruff check` until all checks pass

### 5. Commit with Sign-off (DCO)

All commits must be signed off to comply with the Developer Certificate of Origin (DCO):

```bash
# Stage your changes
git add examples/your_domain/your_example.py
git add examples/your_domain/README.md

# Commit with sign-off
git commit -s --author "Your Name <your.email@example.com>" -m "Add [model-name] support

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
git push origin <branch-name>
```

### 7. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Compare & pull request" for your branch
3. Fill out the PR template with:
   - **Title:** Clear, descriptive title (e.g., "Add Llama-3.2-Vision Support" or "Fix memory leak in KV cache")
   - **Description:** 
     - What changes were made and why
     - What problem it solves or feature it adds
     - Any special considerations or breaking changes
     - Links to relevant documentation, issues, or model cards (if applicable)
   - **Testing:** Describe how you tested your changes

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


Here are a few things you can do that will increase the likelihood of your pull request to be accepted:

- Follow the existing style where possible.
- Write tests.
- Keep your change as focused as possible.
  If you want to make multiple independent changes, please consider submitting them as separate pull requests.
- Write a [good commit message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).
