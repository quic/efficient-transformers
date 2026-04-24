import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def artifacts_dir():
    """Create and return artifacts directory

    Uses NIGHTLY_PIPELINE_ARTIFACTS_DIR env var if set,
    otherwise uses ~/.cache/Nightly_Pipeline/{timestamp}
    """
    env_path = os.environ.get("NIGHTLY_PIPELINE_ARTIFACTS_DIR")

    if env_path:
        artifacts_path = Path(env_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_path = Path.home() / ".cache" / "Nightly_Pipeline" / timestamp

    artifacts_path.mkdir(parents=True, exist_ok=True)
    return artifacts_path


@pytest.fixture(scope="session")
def artifacts_file(artifacts_dir):
    """Path to shared artifacts JSON file"""
    return artifacts_dir / "model_artifacts.json"


def load_artifacts(filepath):
    """Load artifacts from JSON file"""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def save_artifacts(filepath, data):
    """Save artifacts to JSON file, converting non-serializable objects"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def make_serializable(obj):
        """Recursively convert non-serializable objects to serializable format"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, "__dataclass_fields__"):
            # Handle dataclass instances
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        return obj

    serializable_data = make_serializable(data)
    with open(filepath, "w") as f:
        json.dump(serializable_data, f, indent=2)


@pytest.fixture
def model_artifacts(artifacts_file):
    """Fixture to get/set model artifacts"""
    artifacts = load_artifacts(artifacts_file)
    yield artifacts
    # Save after each test
    save_artifacts(artifacts_file, artifacts)


@pytest.fixture
def get_model_config():
    """Load model and pipeline configs"""
    model_config_path = os.path.join(os.path.dirname(__file__), "configs/validated_models.json")
    with open(model_config_path, "r") as f:
        config = json.load(f)

    pipeline_config_path = os.path.join(os.path.dirname(__file__), "configs/pipeline_configs.json")
    with open(pipeline_config_path, "r") as f:
        pipeline_configs = json.load(f)

    return config, pipeline_configs
