# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Static dependency analysis for the regression Jenkins pipeline.

The selector intentionally uses only the Python standard library.  It runs before
QEfficient is installed and fails closed whenever it cannot confidently map a
production change to the tests which exercise it.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

SCHEMA_VERSION = 2

STAGES = (
    "export_compile",
    "qaic_llm",
    "qaic_feature",
    "qaic_multimodal",
    "qaic_reranker",
    "qaic_diffusion",
    "cli",
    "dynamo_qaic",
)

HARD_FULL_FILES = {
    "pyproject.toml",
    "scripts/Jenkinsfile",
    "scripts/JenkinsFileFullCi",
}
IGNORED_FILES = {
    ".gitignore",
    ".pre-commit-config.yaml",
    "CODE-OF-CONDUCT.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README.md",
}
IGNORED_PREFIXES = (".github/", "docs/", "skills_studio/")
OUT_OF_SCOPE_TEST_PREFIXES = ("tests/unit_test/", "tests/nightly_pipeline/", "tests/vllm/")
SELECTIVE_OMITTED_MARKERS = {"full_layers"}

_MODEL_WRAPPER_RE = re.compile(r"^QEfficient/transformers/models/([^/]+)/")
_MODEL_KEYS = {"model", "model_id", "model_name", "model_type", "architecture", "architectures"}
_REFLECTION_NAMES = {"__import__", "compile", "eval", "exec", "getattr", "globals", "locals", "setattr"}
_HOOK_NAMES = {
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_generate_tests",
    "pytest_ignore_collect",
}


@dataclass(frozen=True)
class Change:
    status: str
    old_path: str | None
    new_path: str | None

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


@dataclass
class Symbol:
    key: str
    path: str
    name: str
    kind: str
    digest: str
    dependencies: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    fixtures: set[str] = field(default_factory=set)
    markers: set[str] = field(default_factory=set)
    autouse: bool = False


@dataclass
class Module:
    path: str
    name: str
    symbols: dict[str, Symbol] = field(default_factory=dict)
    imports: dict[str, str] = field(default_factory=dict)
    unsafe_reasons: list[str] = field(default_factory=list)
    module_markers: set[str] = field(default_factory=set)


@dataclass
class TestCase:
    nodeid: str
    symbol: str
    path: str
    markers: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)
    fixtures: set[str] = field(default_factory=set)
    stages: set[str] = field(default_factory=set)


@dataclass
class ImpactPlan:
    mode: str
    base: str
    head: str
    changed_files: list[str]
    reasons: list[str]
    unresolved: list[str]
    stages: dict[str, dict[str, object]]
    selected_symbols: list[str] = field(default_factory=list)
    llm: dict[str, object] | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GitTree:
    """Read a commit without checking it out."""

    def __init__(self, repo: Path, revision: str):
        self.repo = repo
        self.revision = revision

    def files(self, prefixes: Iterable[str] = ()) -> list[str]:
        command = ["git", "ls-tree", "-r", "--name-only", self.revision]
        command.extend(prefixes)
        output = _git(self.repo, command)
        return [line for line in output.splitlines() if line]

    def read(self, path: str) -> str | None:
        process = subprocess.run(
            ["git", "show", f"{self.revision}:{path}"],
            cwd=self.repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return process.stdout if process.returncode == 0 else None


def _git(repo: Path, command: list[str]) -> str:
    return subprocess.check_output(command, cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()


def resolve_changes(repo: Path, base: str, head: str = "HEAD") -> tuple[str, str, list[Change]]:
    base_sha = _git(repo, ["git", "rev-parse", "--verify", f"{base}^{{commit}}"])
    head_sha = _git(repo, ["git", "rev-parse", "--verify", f"{head}^{{commit}}"])
    merge_base = _git(repo, ["git", "merge-base", base_sha, head_sha])
    output = _git(
        repo,
        ["git", "diff", "--name-status", "--find-renames", f"{merge_base}...{head_sha}"],
    )
    changes = []
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) == 3:
            changes.append(Change(status="R", old_path=fields[1], new_path=fields[2]))
        elif len(fields) == 2:
            path = fields[1]
            changes.append(
                Change(
                    status=status[0],
                    old_path=path if status.startswith("D") else None,
                    new_path=None if status.startswith("D") else path,
                )
            )
        else:
            raise ValueError(f"unrecognized git diff record: {line!r}")
    return merge_base, head_sha, changes


def _module_name(path: str) -> str:
    pure = PurePosixPath(path)
    parts = list(pure.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _digest(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    pieces = []
    while isinstance(node, ast.Attribute):
        pieces.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        pieces.append(node.id)
    return ".".join(reversed(pieces))


def _strings(node: ast.AST) -> set[str]:
    values = {
        child.value for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    models = {_normalize_model(value) for value in values if _looks_like_model(value)}
    for child in ast.walk(node):
        if not isinstance(child, ast.Dict):
            continue
        for key, value in zip(child.keys, child.values):
            if not isinstance(key, ast.Constant) or key.value not in _MODEL_KEYS:
                continue
            for item in ast.walk(value):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    models.add(_normalize_model(item.value))
    return models


def _looks_like_model(value: str) -> bool:
    lowered = value.lower()
    model_tokens = ("llama", "qwen", "gemma", "gpt", "mistral", "whisper", "bert", "falcon", "phi")
    return "/" in value or any(token in lowered for token in model_tokens)


def _normalize_model(value: str) -> str:
    value = value.lower().replace("-", "_").replace(".", "_")
    return re.sub(r"[^a-z0-9_/]+", "_", value).strip("_")


def _extract_markers(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> set[str]:
    markers = set()
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        if ".mark." in name:
            markers.add(name.rsplit(".", 1)[-1])
    return markers


def _fixture_metadata(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bool, bool]:
    fixture = False
    autouse = False
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        if name.endswith("fixture"):
            fixture = True
            if isinstance(decorator, ast.Call):
                autouse = any(
                    keyword.arg == "autouse" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                    for keyword in decorator.keywords
                )
    return fixture, autouse


def _resolve_relative(module: str, level: int, imported: str | None, is_package: bool) -> str:
    parts = module.split(".") if is_package else module.split(".")[:-1]
    if level > 1:
        parts = parts[: max(0, len(parts) - level + 1)]
    elif level == 0:
        parts = []
    if imported:
        parts.extend(imported.split("."))
    return ".".join(parts)


def parse_module(path: str, source: str) -> Module:
    module_name = _module_name(path)
    is_package = PurePosixPath(path).name == "__init__.py"
    result = Module(path=path, name=module_name)
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, UnicodeError) as error:
        result.unsafe_reasons.append(f"unparsable Python: {error}")
        return result

    def add_definition(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, qualified_name: str) -> None:
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        fixture = False
        autouse = False
        arguments = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fixture, autouse = _fixture_metadata(node)
            if fixture:
                kind = "fixture"
            arguments = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            arguments.discard("self")
            arguments.discard("cls")
        key = f"{module_name}:{qualified_name}"
        result.symbols[qualified_name] = Symbol(
            key=key,
            path=path,
            name=qualified_name,
            kind=kind,
            digest=_digest(node),
            names={child.id for child in ast.walk(node) if isinstance(child, ast.Name)},
            models=_strings(node),
            fixtures=arguments,
            markers=_extract_markers(node),
            autouse=autouse,
        )
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add_definition(child, f"{qualified_name}.{child.name}")

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                result.imports[local_name] = alias.name
                result.symbols[local_name] = Symbol(
                    key=f"{module_name}:{local_name}",
                    path=path,
                    name=local_name,
                    kind="import",
                    digest=_digest(node),
                    dependencies={alias.name},
                )
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_relative(module_name, node.level, node.module, is_package)
            for alias in node.names:
                if alias.name == "*":
                    result.unsafe_reasons.append("star import")
                    continue
                target = f"{imported_module}:{alias.name}" if imported_module else alias.name
                local_name = alias.asname or alias.name
                result.imports[local_name] = target
                result.symbols[local_name] = Symbol(
                    key=f"{module_name}:{local_name}",
                    path=path,
                    name=local_name,
                    kind="import",
                    digest=f"{_digest(node)}:{alias.name}:{alias.asname}",
                    dependencies={target},
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_definition(node, node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names = [item.id for item in ast.walk(target) if isinstance(item, ast.Name)]
                for name in names:
                    key = f"{module_name}:{name}"
                    result.symbols[name] = Symbol(
                        key=key,
                        path=path,
                        name=name,
                        kind="assignment",
                        digest=_digest(node),
                        names={child.id for child in ast.walk(node) if isinstance(child, ast.Name)},
                        models=_strings(node),
                    )
            if any(name == "pytestmark" for target in targets for name in [getattr(target, "id", None)]):
                result.module_markers.update(
                    child.attr
                    for child in ast.walk(node)
                    if isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Attribute)
                    and child.value.attr == "mark"
                )
        elif isinstance(
            node,
            (
                ast.Expr,
                ast.With,
                ast.AsyncWith,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.If,
                ast.Try,
                ast.Match,
                ast.Raise,
                ast.Assert,
                ast.Delete,
            ),
        ):
            # Docstrings are harmless. Other executable module statements make
            # import ordering and side effects impossible to infer statically.
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                result.unsafe_reasons.append(f"module-level {type(node).__name__}")

    for symbol in result.symbols.values():
        for name in symbol.names:
            if name in result.symbols and name != symbol.name:
                symbol.dependencies.add(result.symbols[name].key)
            owner = symbol.name.rsplit(".", 1)[0] if "." in symbol.name else None
            owned_name = f"{owner}.{name}" if owner else None
            if owned_name in result.symbols and owned_name != symbol.name:
                symbol.dependencies.add(result.symbols[owned_name].key)
            target = result.imports.get(name)
            if target:
                symbol.dependencies.add(target)
        node_source = symbol.digest.lower()
        if "importlib" in node_source or any(f"name(id='{name}'" in node_source for name in _REFLECTION_NAMES):
            result.unsafe_reasons.append(f"reflection or dynamic import in {symbol.name}")
    return result


def build_snapshot(tree: GitTree) -> tuple[dict[str, Module], list[str]]:
    modules = {}
    errors = []
    paths = tree.files(("QEfficient", "tests"))
    for path in paths:
        if not path.endswith(".py"):
            continue
        source = tree.read(path)
        if source is None:
            errors.append(f"could not read {tree.revision}:{path}")
            continue
        module = parse_module(path, source)
        modules[path] = module
    _link_imports(modules)
    return modules, errors


def _link_imports(modules: Mapping[str, Module]) -> None:
    by_name = {module.name: module for module in modules.values()}
    for module in modules.values():
        for symbol in module.symbols.values():
            expanded = set()
            for dependency in symbol.dependencies:
                if ":" in dependency:
                    imported_module, imported_name = dependency.split(":", 1)
                    target_module = by_name.get(imported_module)
                    if target_module and imported_name in target_module.symbols:
                        expanded.add(target_module.symbols[imported_name].key)
                    else:
                        expanded.add(dependency)
                else:
                    target_module = by_name.get(dependency)
                    if target_module:
                        expanded.update(item.key for item in target_module.symbols.values())
            symbol.dependencies.update(expanded)


def _changed_symbols(old: Module | None, new: Module | None) -> set[str]:
    old_symbols = old.symbols if old else {}
    new_symbols = new.symbols if new else {}
    changed = set()
    for name in old_symbols.keys() | new_symbols.keys():
        before = old_symbols.get(name)
        after = new_symbols.get(name)
        if before is None or after is None or before.digest != after.digest:
            changed.add((after or before).key)
    return changed


def _reverse_graph(*snapshots: Mapping[str, Module]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for modules in snapshots:
        for module in modules.values():
            for symbol in module.symbols.values():
                for dependency in symbol.dependencies:
                    reverse[dependency].add(symbol.key)
    return reverse


def _closure(roots: set[str], reverse: Mapping[str, set[str]]) -> set[str]:
    selected = set(roots)
    queue = deque(roots)
    while queue:
        current = queue.popleft()
        for consumer in reverse.get(current, ()):
            if consumer not in selected:
                selected.add(consumer)
                queue.append(consumer)
    return selected


def _stages_for(path: str, markers: set[str]) -> set[str]:
    if markers & SELECTIVE_OMITTED_MARKERS:
        return set()
    if path.startswith(OUT_OF_SCOPE_TEST_PREFIXES) or "qnn" in markers or "finetune" in markers:
        return set()
    if path.startswith("tests/dynamo/"):
        return {"dynamo_qaic"} if "on_qaic" in markers and "nightly" not in markers else set()
    if path == "tests/transformers/models/reranker/test_reranker_mad.py":
        return {"qaic_reranker"}
    stages = set()
    if "diffusion_models" in markers:
        stages.add("qaic_diffusion")
    if "multimodal" in markers:
        stages.add("qaic_multimodal")
    if "cli" in markers:
        stages.add("cli")
    if "on_qaic" in markers and "feature" in markers:
        stages.add("qaic_feature")
    if "llm_model" in markers:
        stages.add("qaic_llm")
    if "on_qaic" not in markers and "finetune" not in markers:
        stages.add("export_compile")
    return stages


def inventory_tests(modules: Mapping[str, Module]) -> dict[str, TestCase]:
    tests = {}
    symbols_by_key = {symbol.key: symbol for module in modules.values() for symbol in module.symbols.values()}

    def dependency_models(symbol: Symbol) -> set[str]:
        models = set(symbol.models)
        pending = list(symbol.dependencies)
        visited = {symbol.key}
        while pending:
            dependency = pending.pop()
            if dependency in visited:
                continue
            visited.add(dependency)
            target = symbols_by_key.get(dependency)
            if target is not None:
                models.update(target.models)
                pending.extend(target.dependencies)
        return models

    for module in modules.values():
        if not module.path.startswith("tests/") or not PurePosixPath(module.path).name.startswith("test_"):
            continue
        for symbol in module.symbols.values():
            test_name = symbol.name.rsplit(".", 1)[-1]
            if symbol.kind != "function" or not test_name.startswith("test_"):
                continue
            owner_name = symbol.name.rsplit(".", 1)[0] if "." in symbol.name else None
            owner = module.symbols.get(owner_name) if owner_name else None
            markers = module.module_markers | symbol.markers | (owner.markers if owner else set())
            nodeid = f"{module.path}::{symbol.name.replace('.', '::')}"
            tests[symbol.key] = TestCase(
                nodeid=nodeid,
                symbol=symbol.key,
                path=module.path,
                markers=markers,
                models=dependency_models(symbol) | (dependency_models(owner) if owner else set()),
                fixtures=set(symbol.fixtures),
                stages=_stages_for(module.path, markers),
            )
    return tests


def _model_terms(path: str, old: Module | None, new: Module | None, changed: set[str]) -> set[str]:
    terms = set()
    match = _MODEL_WRAPPER_RE.match(path)
    if match:
        terms.add(_normalize_model(match.group(1)))
    changed_names = {key.rsplit(":", 1)[-1] for key in changed}
    for module in (old, new):
        if module:
            for name in changed_names:
                symbol = module.symbols.get(name)
                if symbol is None:
                    continue
                terms.update(symbol.models)
                if symbol.kind == "class":
                    terms.add(_normalize_model(symbol.name))
    return {term for term in terms if len(term) > 2}


def _config_terms(source: str) -> set[str]:
    payload = json.loads(source)
    terms = set()

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and (key in _MODEL_KEYS or _looks_like_model(value)):
            terms.add(_normalize_model(value))

    visit(payload)
    return terms


def _term_matches(term: str, candidates: set[str]) -> bool:
    aliases = {term, term.replace("_moe", ""), term.replace("_vl", "")}
    return any(
        alias and (alias == candidate or alias in candidate or candidate in alias)
        for alias in aliases
        for candidate in candidates
    )


def _empty_plan(
    mode: str,
    base: str,
    head: str,
    changes: list[Change],
    reasons: list[str],
    unresolved: list[str],
) -> ImpactPlan:
    return ImpactPlan(
        mode=mode,
        base=base,
        head=head,
        changed_files=sorted({change.path for change in changes}),
        reasons=reasons,
        unresolved=unresolved,
        stages={
            stage: {
                "enabled": mode == "full",
                "nodeids": [],
                "changed_nodeids": [],
                "profile_override_nodeids": [],
            }
            for stage in STAGES
        },
    )


def build_plan(repo: Path, base: str, head: str = "HEAD", force_full: bool = False) -> ImpactPlan:
    repo = repo.resolve()
    merge_base, head_sha, changes = resolve_changes(repo, base, head)
    if force_full:
        return _empty_plan("full", merge_base, head_sha, changes, ["full CI manually forced"], [])
    if not changes:
        return _empty_plan("no_tests", merge_base, head_sha, changes, ["no changed files"], [])

    paths = {change.path for change in changes}
    hard = sorted(path for path in paths if path in HARD_FULL_FILES or path.startswith("scripts/ci_impact/"))
    if hard:
        reasons = [f"unconditional full-CI path: {path}" for path in hard]
        return _empty_plan("full", merge_base, head_sha, changes, reasons, [])

    meaningful = {path for path in paths if path not in IGNORED_FILES and not path.startswith(IGNORED_PREFIXES)}
    if not meaningful:
        return _empty_plan(
            "no_tests",
            merge_base,
            head_sha,
            changes,
            ["changes are not consumed by Jenkins model regression"],
            [],
        )
    if meaningful == {"MANIFEST.in"}:
        return _empty_plan("install_only", merge_base, head_sha, changes, ["packaging manifest changed"], [])

    old_tree = GitTree(repo, merge_base)
    new_tree = GitTree(repo, head_sha)
    old_modules, old_errors = build_snapshot(old_tree)
    new_modules, new_errors = build_snapshot(new_tree)
    if old_errors or new_errors:
        return _empty_plan(
            "full",
            merge_base,
            head_sha,
            changes,
            ["source snapshot generation failed"],
            old_errors + new_errors,
        )

    roots = set()
    reasons = []
    unresolved = []
    model_terms = set()
    changed_test_roots = set()
    known_non_python = set()

    for change in changes:
        path = change.path
        old_module = old_modules.get(change.old_path or path)
        new_module = new_modules.get(change.new_path or path)
        if path == "MANIFEST.in" or path in IGNORED_FILES or path.startswith(IGNORED_PREFIXES):
            known_non_python.add(path)
            continue
        if path == "tests/conftest.py":
            changed = _changed_symbols(old_module, new_module)
            unsafe = []
            for key in changed:
                name = key.rsplit(":", 1)[-1]
                symbol = (new_module and new_module.symbols.get(name)) or (old_module and old_module.symbols.get(name))
                if name in _HOOK_NAMES or (symbol and symbol.autouse):
                    unsafe.append(name)
            if unsafe:
                reason = f"global pytest behavior changed: {', '.join(sorted(unsafe))}"
                return _empty_plan("full", merge_base, head_sha, changes, [reason], [])
        if path.endswith(".py") and (path.startswith("QEfficient/") or path.startswith("tests/")):
            if (old_module and old_module.unsafe_reasons) or (new_module and new_module.unsafe_reasons):
                details = (old_module.unsafe_reasons if old_module else []) + (
                    new_module.unsafe_reasons if new_module else []
                )
                return _empty_plan(
                    "full",
                    merge_base,
                    head_sha,
                    changes,
                    [f"unsafe static analysis for {path}"],
                    sorted(set(details)),
                )
            changed = _changed_symbols(old_module, new_module)
            roots.update(changed)
            model_terms.update(_model_terms(path, old_module, new_module, changed))
            if path.startswith("tests/"):
                changed_test_roots.update(changed)
            reasons.append(f"AST change: {path}")
        elif path.startswith("tests/configs/") and path.endswith(".json"):
            source = new_tree.read(path) or old_tree.read(path) or ""
            try:
                model_terms.update(_config_terms(source))
            except json.JSONDecodeError as error:
                return _empty_plan(
                    "full",
                    merge_base,
                    head_sha,
                    changes,
                    [f"unparsable model inventory: {path}"],
                    [str(error)],
                )
            known_non_python.add(path)
            reasons.append(f"model inventory changed: {path}")
        elif path.startswith("examples/"):
            known_non_python.add(path)
            reasons.append(f"example dependency changed: {path}")
        elif path == "scripts/specializations.json":
            known_non_python.add(path)
            reasons.append("compiler specialization configuration changed")
        else:
            unresolved.append(path)

    if unresolved:
        return _empty_plan(
            "full",
            merge_base,
            head_sha,
            changes,
            ["unclassified production/configuration changes"],
            unresolved,
        )

    reverse = _reverse_graph(old_modules, new_modules)
    selected_symbols = _closure(roots, reverse)
    tests = inventory_tests(new_modules)
    selected_tests = {key for key in tests if key in selected_symbols or key in changed_test_roots}

    # Fixtures are name-bound by pytest rather than imported. Propagate a
    # changed fixture to every eligible test requesting it.
    all_symbols = {
        symbol.key: symbol
        for module in list(old_modules.values()) + list(new_modules.values())
        for symbol in module.symbols.values()
    }
    fixture_names = {
        key.rsplit(":", 1)[-1] for key in selected_symbols if key in all_symbols and all_symbols[key].kind == "fixture"
    }
    autouse_roots = {
        str(PurePosixPath(all_symbols[key].path).parent) + "/"
        for key in selected_symbols
        if key in all_symbols and all_symbols[key].kind == "fixture" and all_symbols[key].autouse
    }
    for key, test in tests.items():
        if fixture_names & test.fixtures or any(test.path.startswith(root) for root in autouse_roots):
            selected_tests.add(key)

    if model_terms:
        for key, test in tests.items():
            if any(_term_matches(term, test.models) for term in model_terms):
                selected_tests.add(key)

    # Explicit non-Python relationships which cannot be represented by import
    # edges in the AST graph.
    for path in known_non_python:
        if path.startswith("tests/configs/"):
            stem = PurePosixPath(path).stem.replace("_configs", "")
            for key, test in tests.items():
                if stem in test.path or any(_term_matches(term, test.models) for term in model_terms):
                    selected_tests.add(key)
        elif path.startswith("examples/dynamo/"):
            selected_tests.update(key for key, test in tests.items() if "dynamo_qaic" in test.stages)
        elif "diffusion" in path or "/flux" in path or "/wan" in path:
            selected_tests.update(key for key, test in tests.items() if "qaic_diffusion" in test.stages)
        elif path == "scripts/specializations.json":
            selected_tests.update(
                key for key, test in tests.items() if test.path.startswith(("tests/cloud/", "tests/base/"))
            )

    stage_nodeids: dict[str, set[str]] = {stage: set() for stage in STAGES}
    changed_nodeids: dict[str, set[str]] = {stage: set() for stage in STAGES}
    for key in selected_tests:
        test = tests[key]
        for stage in test.stages:
            stage_nodeids[stage].add(test.nodeid)
            if key in changed_test_roots:
                changed_nodeids[stage].add(test.nodeid)

    enabled = {stage for stage, nodeids in stage_nodeids.items() if nodeids}
    if enabled == set(STAGES):
        return _empty_plan(
            "full", merge_base, head_sha, changes, ["dependency closure reaches every Jenkins stage"], []
        )

    production_changes = any(path.startswith("QEfficient/") for path in meaningful)
    if production_changes and not enabled:
        return _empty_plan(
            "full",
            merge_base,
            head_sha,
            changes,
            ["production changes produced no confident deterministic test matches"],
            sorted(model_terms),
        )

    mode = "selective" if enabled else ("install_only" if "MANIFEST.in" in meaningful else "no_tests")
    plan = _empty_plan(
        mode,
        merge_base,
        head_sha,
        changes,
        reasons or ["deterministic dependency selection"],
        unresolved,
    )
    plan.selected_symbols = sorted(selected_symbols)
    for stage in STAGES:
        plan.stages[stage] = {
            "enabled": stage in enabled,
            "nodeids": sorted(stage_nodeids[stage]),
            "changed_nodeids": sorted(changed_nodeids[stage]),
            "profile_override_nodeids": sorted(changed_nodeids[stage]),
        }
    return plan


def write_plan(plan: ImpactPlan, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
