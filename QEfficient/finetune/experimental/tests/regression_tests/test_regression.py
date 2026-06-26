import json
import os
import shutil
import subprocess
from typing import List, Optional

import numpy as np
import pytest
import torch
import torch.distributed as dist

# Import sibling test modules so their helpers and classes can be reused
# directly instead of duplicating pipeline-run logic here
import QEfficient.finetune.experimental.tests.test_ddp as ddp_test_module
import QEfficient.finetune.experimental.tests.test_pipeline_parallelism as pp_test_module
from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.tests.test_ddp import STORE_PATH

logger = Logger(__name__)

# ============================================================================
# Config
# ============================================================================

# SDK version string used to name golden files (e.g. "finetuning_pipeline_ddp_SDK_1.22.0.112.json")
BASELINE_SDK_VERSION = "SDK_1.22.0.112"
# Set UPDATE_GOLDEN=1 to regenerate golden files instead of comparing against them
UPDATE_GOLDEN = os.getenv("UPDATE_GOLDEN") == "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(BASE_DIR, "goldens")  # Directory that stores JSON golden files

LOSS_TOL = 1e-3  # Absolute tolerance for loss regression comparisons
OUTPUT_DIR_DDP = os.path.join(BASE_DIR, "tmp", "test_regression_ddp")
OUTPUT_DIR_SINGLE = os.path.join(BASE_DIR, "tmp", "test_regression_single")
OUTPUT_DIR_PP2 = os.path.join(BASE_DIR, "tmp", "test_regression_pp2")
WORLD_RANK = 2  # Number of DDP workers used in regression runs


# ============================================================================
# Golden Utils
# ============================================================================


def _golden_path(name: str) -> str:
    """Return the absolute path for a golden JSON file identified by ``name``."""
    return os.path.join(GOLDEN_DIR, f"{name}_{BASELINE_SDK_VERSION}.json")


def save_golden(name: str, data: dict):
    """Persist ``data`` as a JSON golden file for future regression comparisons."""
    try:
        os.makedirs(GOLDEN_DIR, exist_ok=True)  # Create goldens/ dir if it doesn't exist
        path = _golden_path(name)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"✅ Saved golden: {path}")
    except Exception as e:
        logger.error(f"Failed to save golden {name}: {e}")
        raise


def load_golden(name: str) -> Optional[dict]:
    """Load a previously saved golden file; returns None if the file is missing."""
    try:
        path = _golden_path(name)

        if not os.path.exists(path):
            logger.warning(f"Missing golden: {path}")
            return None  # Caller will treat this as "no baseline yet"

        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load golden {name}: {e}")
        return None


# ============================================================================
# Get free devices to avoid QAIC error
# Device Availability Utils
# ============================================================================


def _parse_qaic_device_info() -> dict:
    """
    Parse qaic-util output and return a dict of device info.
    Returns:
        dict: {qid: {"nsp_total": int, "nsp_free": int, "status": str}, ...}
    """
    devices = {}
    try:
        result = subprocess.run(
            ["qaic-util"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        current_qid = None
        current_info = {}

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("QID "):
                # Flush the previous device entry before starting a new one
                if current_qid is not None:
                    devices[current_qid] = current_info
                try:
                    current_qid = int(line.split("QID ")[1].strip())
                    current_info = {"nsp_total": 0, "nsp_free": 0, "status": "Unknown"}
                except ValueError:
                    current_qid = None
                    current_info = {}
            elif current_qid is not None:
                # Parse per-device fields from the indented lines below "QID N"
                if line.startswith("Status:"):
                    current_info["status"] = line.split("Status:")[1].strip()
                elif line.startswith("Nsp Total:"):
                    try:
                        current_info["nsp_total"] = int(line.split("Nsp Total:")[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("Nsp Free:"):
                    try:
                        current_info["nsp_free"] = int(line.split("Nsp Free:")[1].strip())
                    except ValueError:
                        pass

        # Flush the last device entry after the loop ends
        if current_qid is not None:
            devices[current_qid] = current_info

    except subprocess.TimeoutExpired:
        logger.warning("qaic-util timed out while checking device availability")
    except FileNotFoundError:
        logger.warning("qaic-util not found, skipping device availability check")
    except Exception as e:
        logger.warning(f"Unexpected error running qaic-util: {e}")

    return devices


def _get_free_devices(candidate_ids: list, required_nsps: int = 16, num_devices: int = 2) -> list:
    """
    From a list of candidate device IDs, return up to `num_devices` that are
    fully free (Nsp Free == Nsp Total == required_nsps and Status == Ready).

    Args:
        candidate_ids:  List of QIDs to check (from QAIC_VISIBLE_DEVICES)
        required_nsps:  Minimum free NSPs required (default: 16)
        num_devices:    Number of free devices needed (default: 2)

    Returns:
        List of free device IDs (up to num_devices)
    """
    device_info = _parse_qaic_device_info()
    free_devices = []

    for qid in candidate_ids:
        info = device_info.get(qid)
        if info is None:
            logger.warning(f"QID {qid} not found in qaic-util output")
            continue

        is_ready = info.get("status") == "Ready"
        nsp_total = info.get("nsp_total", 0)
        nsp_free = info.get("nsp_free", 0)
        # A device is considered free only when ALL NSPs are available
        is_free = nsp_free == nsp_total == required_nsps

        logger.info(
            f"QID {qid}: status={info.get('status')}, "
            f"nsp_total={nsp_total}, nsp_free={nsp_free}, "
            f"eligible={is_ready and is_free}"
        )

        if is_ready and is_free:
            free_devices.append(qid)

        # Stop early once we have enough devices
        if len(free_devices) == num_devices:
            break

    return free_devices


def _get_candidate_device_ids() -> list:
    """
    Read QAIC_VISIBLE_DEVICES env var and return list of integer device IDs.

    Returns:
        List of device IDs, e.g. [22, 23, 24, 25]
    """
    visible = os.getenv("QAIC_VISIBLE_DEVICES", "")
    if not visible:
        logger.warning("QAIC_VISIBLE_DEVICES is not set")
        return []
    try:
        # Split on commas and convert each token to int
        return [int(x.strip()) for x in visible.split(",") if x.strip()]
    except ValueError as e:
        logger.warning(f"Failed to parse QAIC_VISIBLE_DEVICES='{visible}': {e}")
        return []


# ============================================================================
# Helpers
# ============================================================================


def extract_losses(loss_list):
    """Extract loss values from (step, loss) tuples."""
    return [loss for _, loss in loss_list]


def compute_diff(curr: List[float], base: List[float]):
    """Return (max_diff, avg_diff) between two equal-length loss sequences."""
    diffs = [abs(c - b) for c, b in zip(curr, base)]
    return max(diffs), float(np.mean(diffs))


def compare(name: str, curr: List[float], base: List[float], tol: float):
    """Assert that the average difference between ``curr`` and ``base`` is within ``tol``."""
    assert len(curr) == len(base), f"{name} length mismatch"

    max_diff, avg_diff = compute_diff(curr, base)

    logger.info(f"{name}: max={max_diff:.6f}, avg={avg_diff:.6f}, tol={tol}")

    assert avg_diff < tol, f"{name} regression: {avg_diff} > {tol}"


def cleanup():
    """Tear down the distributed process group and release device resources."""
    try:
        # Destroy the process group so QAIC NSPs are released for the next test
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.info("Process group destroyed successfully")
            except Exception as e:
                logger.warning(f"Error destroying process group: {e}")
        # Release CUDA memory if a CUDA device is also present
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Attempt to reset QAIC device state (no-op if torch_qaic is absent)
        try:
            import torch_qaic

            if hasattr(torch_qaic, "qaic"):
                # Reset device state
                pass
        except (ImportError, Exception):
            pass

        # Clean up temporary output directories
        for output_dir in [OUTPUT_DIR_DDP, OUTPUT_DIR_SINGLE, OUTPUT_DIR_PP2]:
            try:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                    logger.info(f"Removed output directory: {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove directory {output_dir}: {e}")

        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")


# ============================================================================
# Regression Tests (DRIVEN BY PIPELINE PARITY)
# ============================================================================


@pytest.mark.skipif(os.getenv("RUN_REGRESSION") != "1", reason="Run only when RUN_REGRESSION=1")
class TestPipelineRegression:
    def setup_method(self):
        """
        Before each test:
        - Read the 4 candidate devices from QAIC_VISIBLE_DEVICES
        - Find 2 fully free devices among them
        - Override QAIC_VISIBLE_DEVICES with those 2 free devices
        - Skip the test if fewer than 2 free devices are available
        """
        candidate_ids = _get_candidate_device_ids()

        # Need at least 2 candidates to pick 2 free devices from
        if len(candidate_ids) < 2:
            pytest.skip(f"Need at least 2 candidate devices in QAIC_VISIBLE_DEVICES, got: {candidate_ids}")

        logger.info(f"Candidate devices from QAIC_VISIBLE_DEVICES: {candidate_ids}")

        # Filter candidates down to those that are fully idle
        free_devices = _get_free_devices(candidate_ids, required_nsps=16, num_devices=WORLD_RANK)

        if len(free_devices) < 2:
            pytest.skip(
                f"Not enough free QAIC devices. "
                f"Need 2 with 16 free NSPs each, "
                f"found {len(free_devices)} from candidates {candidate_ids}. "
                f"Free devices: {free_devices}"
            )

        # Narrow the visible device set so the pipeline only uses the 2 free devices
        selected = ",".join(str(d) for d in free_devices)
        os.environ["QAIC_VISIBLE_DEVICES"] = selected
        logger.info(f"Selected free devices for regression test: {selected}")

    def _run_pipeline(self):
        """
        - Run pipeline parity test ONLY ONCE
        - Reuse returned losses for regression
        - Properly clean up resources between runs
        """
        single_loss = None
        ddp_loss = None
        pp2_loss = None

        try:
            # Remove any stale FileStore lock from a previous run
            if os.path.exists(STORE_PATH):
                try:
                    os.remove(STORE_PATH)
                    logger.info(f"Cleaned up FileStore at {STORE_PATH}")
                except Exception as e:
                    logger.warning(f"Failed to clean FileStore: {e}")

            # --- DDP pipeline run ---
            try:
                logger.info("Running DDP pipeline...")
                # Allocate a fresh port to avoid conflicts with other tests
                port = ddp_test_module.TestDDPPipelineParity._get_unique_port()
                os.environ["MASTER_PORT"] = str(port)

                ddp_config = ddp_test_module.TestDDPPipelineParity()._build_config_dict(
                    backend="qccl",
                    output_dir=OUTPUT_DIR_DDP,
                )
                ddp_loss, _ = ddp_test_module.TestDDPPipelineParity()._run_ddp(ddp_config, port=port)
                logger.info(f"DDP pipeline completed with {len(ddp_loss) if ddp_loss else 0} loss entries")
            except Exception as e:
                logger.error(f"DDP pipeline error: {e}")
                logger.log_exception("DDP pipeline error details:")
                raise
            finally:
                cleanup()  # Release DDP resources before the next run

            # --- Single-device pipeline run ---
            try:
                logger.info("Running single-device pipeline...")
                single_config = ddp_test_module.TestDDPPipelineParity()._build_config_dict(
                    backend=None,
                    output_dir=OUTPUT_DIR_SINGLE,
                )
                single_loss, _ = ddp_test_module.TestDDPPipelineParity()._run_single(single_config)
                logger.info(
                    f"Single-device pipeline completed with {len(single_loss) if single_loss else 0} loss entries"
                )
            except Exception as e:
                logger.error(f"Single-device pipeline error: {e}")
                raise
            finally:
                cleanup()  # Release resources before the PP run

            # --- Pipeline Parallelism run ---
            try:
                logger.info("Running pipeline parallelism pipeline...")
                pp_config = pp_test_module.TestPPPipelineParity()._build_config_manager(
                    pp_degree=2,
                    output_dir=OUTPUT_DIR_PP2,
                )
                pp2_loss, _ = pp_test_module.TestPPPipelineParity()._run_pipeline(pp_config)
                logger.info(f"Pipeline parallelism completed with {len(pp2_loss) if pp2_loss else 0} loss entries")
            except Exception as e:
                logger.error(f"Pipeline parallelism error: {e}")
                raise
            finally:
                cleanup()  # Final resource release

            return single_loss, ddp_loss, pp2_loss

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    # ======================================================================

    def test_regression_pipeline_losses(self):
        """
        - Main regression entrypoint
        - Uses pipeline parity test to generate results
        - Saves OR compares against golden
        - Properly handles errors and cleanup
        """
        try:
            single_loss, ddp_loss, pp2_loss = self._run_pipeline()
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            logger.log_exception("Pipeline execution error details:")
            raise
        finally:
            # Guarantee cleanup even when _run_pipeline raises
            cleanup()

        # All three training modes must have produced loss data
        assert single_loss is not None, "Single-device loss is None"
        assert ddp_loss is not None, "DDP loss is None"
        assert pp2_loss is not None, "Pipeline parallelism loss is None"

        # Bundle current results with their corresponding golden keys
        data = {
            "single": single_loss,
            "single_golden": "finetuning_pipeline_single",
            "ddp": ddp_loss,
            "ddp_golden": "finetuning_pipeline_ddp",
            "pp": pp2_loss,
            "pp_golden": "finetuning_pipeline_pp2",
        }

        if UPDATE_GOLDEN:
            # UPDATE_GOLDEN=1: overwrite existing baselines with current results
            logger.info("UPDATE_GOLDEN=1 → saving new baseline")
            save_golden(data["single_golden"], data["single"])
            save_golden(data["ddp_golden"], data["ddp"])
            save_golden(data["pp_golden"], data["pp"])
            return

        golden_single = load_golden("finetuning_pipeline_single")
        golden_ddp = load_golden("finetuning_pipeline_ddp")
        golden_pp = load_golden("finetuning_pipeline_pp2")

        if golden_single is None or golden_ddp is None or golden_pp is None:
            # No baseline exists yet — create one from the current run
            logger.info("No golden found → creating new baseline")
            save_golden(data["single_golden"], data["single"])
            save_golden(data["ddp_golden"], data["ddp"])
            save_golden(data["pp_golden"], data["pp"])
            return

        # ------------------------------------------------------------------
        # Compare single-device losses against the stored golden
        # ------------------------------------------------------------------
        try:
            curr_single = extract_losses(single_loss)
            base_single = extract_losses(golden_single)
            compare("Single Loss", curr_single, base_single, LOSS_TOL)
            logger.info("✅ Single-device loss comparison passed")
        except AssertionError as e:
            logger.error(f"Single-device loss comparison failed: {e}")
            raise

        # ------------------------------------------------------------------
        # Compare DDP losses against the stored golden
        # ------------------------------------------------------------------
        try:
            curr_ddp = extract_losses(ddp_loss)
            base_ddp = extract_losses(golden_ddp)
            compare("DDP Loss", curr_ddp, base_ddp, LOSS_TOL)
            logger.info("✅ DDP loss comparison passed")
        except AssertionError as e:
            logger.error(f"DDP loss comparison failed: {e}")
            raise

        # ------------------------------------------------------------------
        # Compare Pipeline Parallelism losses against the stored golden
        # ------------------------------------------------------------------
        try:
            curr_pp = extract_losses(pp2_loss)
            base_pp = extract_losses(golden_pp)
            compare("PP Loss", curr_pp, base_pp, LOSS_TOL)
            logger.info("Pipeline parallelism loss comparison passed")
        except AssertionError as e:
            logger.error(f"Pipeline parallelism loss comparison failed: {e}")
            raise

        logger.info("All regression tests passed!")
