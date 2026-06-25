# Documentation

This directory contains the documentation for QEfficient, built with [MkDocs](https://www.mkdocs.org/) and auto-deployed to GitHub Pages via GitHub Actions.

---

## Quick start for contributors

When adding a new feature, run the docs locally to verify everything renders correctly before raising a PR.

### 1. Install dependencies

```bash
pip install "mkdocs>=1.6" "mkdocs-material>=9.5" "mkdocstrings[python]>=0.25" "mike>=2.0"
pip install -e . --no-deps
```

### 2. Live preview (auto-reloads on file save)

```bash
mkdocs serve
```

Open **http://127.0.0.1:8000** in your browser. Any change to a `.md` file or `mkdocs.yml` reloads the page automatically.

### 3. Strict build — catch broken links and warnings

```bash
mkdocs build --strict
```

Strict mode treats all warnings as errors. This is the same check the CI runs on every PR. Fix any errors before pushing.

### 4. Preview versioned docs (optional)

```bash
mike serve
```

Shows all deployed versions with the version switcher, served from the local `gh-pages` branch.

---

## How automated deployment works

No manual deployment steps are ever needed. Everything is handled by the GitHub Actions workflows in `.github/workflows/`:

| Event | Workflow | Action |
|---|---|---|
| PR opened targeting `main` | `docs-check.yml` | Builds docs + runs broken link checker. PR is blocked if anything fails. |
| Merge to `main` | `docs-deploy.yml` | Deploys `main` version to GitHub Pages automatically. |
| Push to `release/v*` branch | `docs-release.yml` | New version auto-appears in the version switcher. |
| Tag `v*` pushed | `docs-release.yml` | Deploys as `stable` (the default version users land on). |
| Delete `release/v*` branch | `docs-release.yml` | Version auto-removed from the switcher. |

---

## Adding API documentation

API reference pages use [mkdocstrings](https://mkdocstrings.github.io/) to auto-generate docs from Python docstrings.

**Step 1** — Add a Google-style docstring to your class or method:

```python
class MyNewClass:
    """
    Brief description of the class.

    Args:
        model_name (str): The HuggingFace model identifier.
        num_cores (int): Number of cores to use.

    Example:
        ```python
        obj = MyNewClass("gpt2", num_cores=16)
        obj.compile()
        ```
    """
```

**Step 2** — Add a `:::` entry in the relevant `docs/source/*.md` file:

```markdown
## `MyNewClass` { #MyNewClass }

::: QEfficient.path.to.MyNewClass
    options:
      members:
        - from_pretrained
        - export
        - compile
        - generate
```

**Step 3** — Run `mkdocs serve` and verify the rendered output at `http://127.0.0.1:8000`.

---

## Directory structure

```
docs/
├── index.md                    # Home page — mirrors the original Sphinx TOC
├── source/                     # All content pages
│   ├── introduction.md
│   ├── quick_start.md
│   ├── features_enablement.md
│   ├── installation.md
│   ├── validate.md
│   ├── supported_features.md
│   ├── release_docs.md
│   ├── qeff_autoclasses.md     # Auto Classes API reference
│   ├── cli_api.md              # CLI API reference
│   ├── diffuser_classes.md     # Diffuser Classes API reference
│   ├── finetune.md
│   ├── blogs.md
│   └── reference.md
├── _static/
│   ├── my_theme.css            # Custom CSS (version selector styling, table fixes)
│   └── cleanup_docstrings.js  # Removes orphaned RST code-block artifacts
├── image/                      # Images used in docs
├── README.md                   # This file
│
│   ── Legacy Sphinx files (kept for reference, not used by MkDocs) ──
├── conf.py
├── index.rst
├── requirements.txt
└── _templates/
```

---

## Legacy Sphinx docs (kept for reference)

The old Sphinx build still works if needed:

```bash
pip install -r docs/requirements.txt
cd docs/
sphinx-build -M html . build
cd build/html && python -m http.server
# Visit http://localhost:8080
```
