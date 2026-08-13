# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Utility for writing a small JSON manifest that captures the installed
QEfficient library, runtime environment, model wrapper, and key dependency
versions at model-construction time.
It'll dump the config for every new model that's run in the current session so multiple component based models would have multiple dump files for each component.

Public API
----------
dump_qeff_library_once(model_architecture, model_name) -> Optional[Path]
    Write (at most once per model per Python process) a manifest under:
        QEFF_HOME / <arch_or_name> / <model_name> / qeff_library-<run_id>.json
    Returns the Path on success, None on any failure.
"""

import json
import logging
import os
import platform
import subprocess
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Optional

import QEfficient
from QEfficient.utils.cache import QEFF_HOME

logger = logging.getLogger(__name__)

# Schema version — bump when the manifest layout changes in a breaking way.
_SCHEMA_VERSION = 1


# Process-level registry: model_name -> Path of the already-written manifest.
# Ensures at most one manifest per model per Python process.
_written: dict[str, Path] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _package_version(package_name: str) -> Optional[str]:
    """Return the installed version of *package_name*, or None if not found."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _all_installed_packages() -> dict:
    """
    Return a mapping of all installed package names to versions.

    This is a lightweight, in-process equivalent of a `pip list` snapshot.
    Names are normalized to the distribution metadata name.
    """
    packages = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.name
        if not name:
            continue
        version = dist.version
        packages[name] = version
    # Stable ordering for deterministic JSON output
    return dict(sorted(packages.items(), key=lambda kv: kv[0].lower()))


def _qeff_version() -> Optional[str]:
    return _package_version("QEfficient")


def _qeff_import_path() -> str:
    return str(Path(QEfficient.__file__).parent)


def _qeff_editable() -> bool:
    """
    Best-effort check: True when QEfficient appears to be installed in
    editable / local-source mode.

    Strategy (in order):
    1. Look for a direct_url.json dist-info entry that carries
       ``{"dir_info": {"editable": true}}``.
    2. Fall back to checking whether the import path sits outside the
       standard site-packages tree (i.e. the source checkout is on sys.path
       directly).
    """
    try:
        dist = importlib_metadata.distribution("QEfficient")
        # PEP 610 — direct_url.json is present for editable installs made with
        # modern pip (>= 21.3).
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            import json as _json

            info = _json.loads(direct_url_text)
            dir_info = info.get("dir_info", {})
            if dir_info.get("editable", False):
                return True
    except Exception:
        pass

    # Heuristic fallback: editable installs typically live outside site-packages.
    import_path = _qeff_import_path()
    return "site-packages" not in import_path


def _run_id() -> str:
    """Build a run identifier from the current timestamp and process ID."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-pid{os.getpid()}"


_QAIC_VERSION_UTIL = "/opt/qti-aic/tools/qaic-version-util"


def _qaic_sdk_versions() -> dict:
    """
    Run qaic-version-util and parse the apps/platform SDK version strings.

    Returns a dict with keys ``apps`` and ``platform``, each set to the
    version string (e.g. ``"AIC.1.23.0.18"``) or ``None`` when the tool is
    unavailable, the entry is missing, or the value is ``"not found"``.
    """
    result = {"apps": None, "platform": None}
    try:
        proc = subprocess.run(
            [_QAIC_VERSION_UTIL],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("apps:"):
                value = line.split(":", 1)[1].strip()
                result["apps"] = value if value != "not found" else None
            elif line.startswith("platform:"):
                value = line.split(":", 1)[1].strip()
                result["platform"] = value if value != "not found" else None
    except Exception:
        pass
    return result


def _build_manifest(model_architecture: Optional[str], model_name: str) -> dict:
    """Assemble the full manifest dictionary."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "model_name": model_name,
            "model_architecture": model_architecture,
        },
        "qeff": {
            "name": "QEfficient",
            "version": _qeff_version(),
            "import_path": _qeff_import_path(),
            "editable": _qeff_editable(),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "qaic_sdk": _qaic_sdk_versions(),
        "dependencies": _all_installed_packages(),
    }


def _manifest_path(model_architecture: Optional[str], model_name: str, run_id: str) -> Path:
    """
    Return the target path for the manifest file.

    Layout: QEFF_HOME / <arch_or_name> / <model_name> / qeff_library-<run_id>.json
    """
    arch_dir = model_architecture if model_architecture else model_name
    return QEFF_HOME / arch_dir / model_name / f"qeff_library-{run_id}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dump_qeff_library_once(model_architecture: Optional[str], model_name: str) -> Optional[Path]:
    """
    Write a QEff library manifest for *model_name* at most once per process.

    If the same *model_name* has already been dumped in this Python run, the
    previously written path is returned immediately without touching the
    filesystem.  A different *model_name* always gets its own manifest under
    its own cache sub-directory.

    Args:
        model_architecture: Architecture string from the model config
            (e.g. ``"LlamaForCausalLM"``).  Pass ``None`` when unavailable.
        model_name: The QEff wrapper model name (e.g. ``"LlamaForCausalLM"``).

    Returns:
        Path to the written (or previously written) manifest, or ``None`` if
        the dump failed for any reason.
    """
    if model_name in _written:
        return _written[model_name]

    try:
        run_id = _run_id()
        path = _manifest_path(model_architecture, model_name, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest = _build_manifest(model_architecture, model_name)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        _written[model_name] = path
        logger.debug("QEff library manifest written: %s", path)
        return path
    except Exception as exc:
        logger.warning("dump_qeff_library_once failed (non-fatal): %s", exc)
        return None
