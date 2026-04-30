# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.ut

from pathlib import Path

import numpy as np
import pytest

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux is the supported environment here.
    fcntl = None


PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parents[1]
PIPELINE_CONFIG_PATH = PIPELINE_ROOT / "configs" / "pipeline_configs.json"


def _load_pipeline_configs():
    with open(PIPELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_pipeline_output_dir():
    """Resolve one stable artifacts directory for all nightly pipeline workers."""
    env_path = os.environ.get("NIGHTLY_PIPELINE_ARTIFACTS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    pipeline_configs = _load_pipeline_configs()
    output_dir = pipeline_configs["causal_pipeline_configs"][0].get("output_dir")
    if output_dir:
        output_path = Path(output_dir).expanduser()
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        return output_path

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return (Path.home() / ".cache" / "Nightly_Pipeline" / timestamp).resolve()


def pytest_configure(config):
    """Compute one shared artifacts path and propagate it to xdist workers."""
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        config._nightly_pipeline_artifacts_dir = Path(workerinput["nightly_pipeline_artifacts_dir"])
        return

    config._nightly_pipeline_artifacts_dir = _resolve_pipeline_output_dir()


def pytest_configure_node(node):
    node.workerinput["nightly_pipeline_artifacts_dir"] = str(node.config._nightly_pipeline_artifacts_dir)


def _lock_path(filepath):
    filepath = Path(filepath)
    return filepath.with_suffix(filepath.suffix + ".lock")


@contextmanager
def _locked_file(lockfile_path):
    lockfile_path = Path(lockfile_path)
    lockfile_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lockfile_path, "a+", encoding="utf-8") as lockfile:
        if fcntl is not None:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)


def _make_serializable(obj):
    """Recursively convert artifact values to JSON-serializable data."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return obj


def _merge_artifacts(existing_data, new_data):
    """Merge per-model payloads without dropping updates from other workers."""
    merged = dict(existing_data)
    for model_name, model_payload in new_data.items():
        if isinstance(model_payload, dict) and isinstance(merged.get(model_name), dict):
            merged[model_name] = {**merged[model_name], **model_payload}
        else:
            merged[model_name] = model_payload
    return merged


@pytest.fixture(scope="session")
def artifacts_dir(request):
    """Create and return one artifacts directory shared by all xdist workers."""
    artifacts_path = request.config._nightly_pipeline_artifacts_dir
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return artifacts_path


@pytest.fixture(scope="session")
def causal_model_artifacts_file(artifacts_dir):
    """Path to shared artifacts JSON file."""
    return artifacts_dir / "causal_model_artifacts.json"


@pytest.fixture(scope="session")
def embedding_model_artifacts_file(artifacts_dir):
    """Path to shared artifacts JSON file."""
    return artifacts_dir / "embedding_model_artifacts.json"


@pytest.fixture(scope="session")
def sequence_model_artifacts_file(artifacts_dir):
    """Path to shared artifacts JSON file."""
    return artifacts_dir / "sequence_model_artifacts.json"


@pytest.fixture(scope="session")
def audio_model_artifacts_file(artifacts_dir):
    """Path to shared artifacts JSON file."""
    return artifacts_dir / "audio_model_artifacts.json"


@pytest.fixture(scope="session")
def audio_embedding_model_artifacts_file(artifacts_dir):
    """Path to shared artifacts JSON file."""
    return artifacts_dir / "audio_embedding_model_artifacts.json"


@pytest.fixture(scope="session")
def image_text_to_text_model_artifacts_file(artifacts_dir):
    """Path to shared artifacts JSON file."""
    return artifacts_dir / "image_text_to_text_model_artifacts.json"


def load_artifacts(filepath):
    """Load artifacts from JSON file."""
    filepath = Path(filepath)
    if filepath.exists() and filepath.stat().st_size > 0:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_artifacts(filepath, data):
    """Atomically merge and save artifacts under a file lock."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = _make_serializable(data)

    with _locked_file(_lock_path(filepath)):
        merged_data = _merge_artifacts(load_artifacts(filepath), serializable_data)
        tmp_path = filepath.with_suffix(filepath.suffix + f".{os.getpid()}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2)
        os.replace(tmp_path, filepath)

    return merged_data


@pytest.fixture
def causal_model_artifacts(causal_model_artifacts_file):
    """Fixture to get/set model artifacts."""
    artifacts = load_artifacts(causal_model_artifacts_file)
    yield artifacts
    save_artifacts(causal_model_artifacts_file, artifacts)


@pytest.fixture
def embedding_model_artifacts(embedding_model_artifacts_file):
    """Fixture to get/set model artifacts."""
    artifacts = load_artifacts(embedding_model_artifacts_file)
    yield artifacts
    save_artifacts(embedding_model_artifacts_file, artifacts)


@pytest.fixture
def sequence_model_artifacts(sequence_model_artifacts_file):
    """Fixture to get/set model artifacts."""
    artifacts = load_artifacts(sequence_model_artifacts_file)
    yield artifacts
    save_artifacts(sequence_model_artifacts_file, artifacts)


@pytest.fixture
def audio_model_artifacts(audio_model_artifacts_file):
    """Fixture to get/set model artifacts."""
    artifacts = load_artifacts(audio_model_artifacts_file)
    yield artifacts
    save_artifacts(audio_model_artifacts_file, artifacts)


@pytest.fixture
def audio_embedding_model_artifacts(audio_embedding_model_artifacts_file):
    """Fixture to get/set model artifacts."""
    artifacts = load_artifacts(audio_embedding_model_artifacts_file)
    yield artifacts
    save_artifacts(audio_embedding_model_artifacts_file, artifacts)


@pytest.fixture
def image_text_to_text_model_artifacts(image_text_to_text_model_artifacts_file):
    """Fixture to get/set model artifacts."""
    artifacts = load_artifacts(image_text_to_text_model_artifacts_file)
    yield artifacts
    save_artifacts(image_text_to_text_model_artifacts_file, artifacts)


@pytest.fixture
def get_model_config():
    """Load model and pipeline configs."""
    model_config_path = PIPELINE_ROOT / "configs" / "validated_models.json"
    with open(model_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    pipeline_configs = _load_pipeline_configs()
    return config, pipeline_configs
