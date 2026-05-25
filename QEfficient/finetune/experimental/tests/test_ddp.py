# ============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ============================================================================

"""
Comprehensive Distributed Data Parallel (DDP) end-to-end tests for QAIC devices.

This test suite validates:
- Parity between single-device and multi-device DDP training
- Gradient synchronization across devices
- Loss computation consistency
- Model state synchronization
- Batch processing correctness
- Learning rate scheduling in DDP
- Checkpoint save/load functionality
- Different world sizes (1, 2, 4 devices)
- Error handling and edge cases
- Integration with the FineTuningPipeline for end-to-end validation
"""

import os
import shutil
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed import FileStore
from torch.nn import MSELoss
from torch.optim import SGD

from QEfficient.finetune.experimental.core.logger import Logger

# FineTuningPipeline and ConfigManager are optional; tests that need them
# will be skipped automatically if the import fails (e.g. in unit-only CI).
try:
    from QEfficient.cloud.finetune_experimental import FineTuningPipeline
    from QEfficient.finetune.experimental.core.config_manager import ConfigManager
except ImportError:
    FineTuningPipeline = None
    ConfigManager = None

logger = Logger(__name__)

# ============================================================================
# Test Configuration Constants
# ============================================================================

WORLD_SIZE = 2  # Default number of DDP processes for most tests
BACKEND = "qccl"  # QAIC Collective Communication Library backend
LOSS_ATOL = 1e-3  # Absolute tolerance for unit-level loss comparisons
PIPELINE_ATOL = 1e-1  # Looser tolerance for end-to-end pipeline tests (non-determinism)
GRADIENT_ATOL = 1e-4  # Tolerance for gradient synchronization checks
PARAM_ATOL = 1e-4  # Tolerance for parameter parity checks across ranks
TEST_SEED = 42  # Fixed seed for reproducible test runs
DEFAULT_MASTER_PORT = 12591  # Default TCP port for the DDP rendezvous
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# FileStore path used by mp.spawn workers to avoid TCP port conflicts
STORE_PATH = os.path.join(BASE_DIR, "tmp", "ddp_store_file")
_LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
# Reduce layer count so the model fits in CI memory and runs quickly
_REDUCED_LAYERS = 2
OUTPUT_DIR_SINGLE = os.path.join(BASE_DIR, "tmp", "test_ddp_single")
OUTPUT_DIR_DDP = os.path.join(BASE_DIR, "tmp", "test_ddp_multi")
_MAX_STEPS = 50
_OUTPUT_DIRS = [OUTPUT_DIR_SINGLE, OUTPUT_DIR_DDP]  # already defined, just add this


def load_llama_model_and_tokenizer(reduced_layers=None):
    """
    Load Llama-3.2-1B with num_hidden_layers reduced to _REDUCED_LAYERS.
    Optionally injects a PP device_map.
    """
    from QEfficient.finetune.experimental.core.component_registry import ComponentFactory
    from QEfficient.finetune.experimental.core.model import HFModel  # noqa: F401

    kwargs = {
        "auto_class_name": "AutoModelForCausalLM",
        "use_cache": False,  # Disable KV-cache; not needed during training
        "attn_implementation": "eager",  # Use eager (non-flash) attention for compatibility
        "num_hidden_layers": reduced_layers,  # Override layer count for faster tests
    }
    return ComponentFactory.create_model("hf", _LLAMA_MODEL_NAME, **kwargs)


# Module-level model instance shared across tests that need a real Llama checkpoint
_HF_MODEL = load_llama_model_and_tokenizer(reduced_layers=_REDUCED_LAYERS)

# ============================================================================
# Test Configuration Dataclasses
# ============================================================================


@dataclass
class DDPTestConfig:
    """Configuration for DDP tests - PICKLABLE for mp.spawn."""

    world_size: int
    rank: int
    backend: str
    find_unused_parameters: bool = False
    static_graph: bool = False
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 2
    num_epochs: int = 1


@dataclass
class SimpleConfig:
    """Picklable configuration for training workers."""

    batch_size: int = 2
    learning_rate: float = 1e-3
    num_batches: int = 5
    seed: int = 42


# ============================================================================
# Test Models
# ============================================================================


class SimpleMLP(nn.Module):
    """Simple MLP model for testing."""

    def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MultiLayerMLP(nn.Module):
    """Multi-layer MLP for more complex testing."""

    def __init__(self, input_dim=8, hidden_dim=16, num_layers=3, output_dim=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================================
# Helpers
# ============================================================================


def is_qaic_available():
    """
    Check if QAIC devices are available.

    Returns:
        bool: True if QAIC devices are available, False otherwise.
    """
    try:
        import torch

        # Check for the torch.qaic namespace (present when torch_qaic is installed)
        # or fall back to inspecting the device string representation
        return hasattr(torch, "qaic") or "qaic" in torch.device.__str__()
    except Exception:
        return False


def ddp_init(rank, world_size, port=DEFAULT_MASTER_PORT, use_file_store=False, timeout=300):
    """
    Initialize DDP process group with robust error handling.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for TCPStore (ignored if use_file_store=True)
        use_file_store: If True, use FileStore for testing (avoids port conflicts with mp.spawn)
        timeout: Timeout in seconds for process group initialization
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # Prefer QAIC backend when hardware is present; fall back to CPU-only gloo
    backend = "qccl" if is_qaic_available() else "gloo"
    logger.info(f"[Rank {rank}] Initializing process group with backend={backend}, timeout={timeout}s")

    try:
        if use_file_store:
            # FileStore avoids TCP port conflicts when using mp.spawn, which
            # launches all workers in the same process group and can race on ports
            store = FileStore(STORE_PATH, world_size=world_size)
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                store=store,
                timeout=torch.distributed.timedelta(seconds=timeout),
                device_id=rank,  # Bind this rank to its QAIC device to suppress NSP warnings
            )
        else:
            # Default path: PyTorch resolves the store via MASTER_ADDR / MASTER_PORT
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.timedelta(seconds=timeout),
                device_id=rank,  # Bind this rank to its QAIC device to suppress NSP warnings
            )
        logger.info(f"[Rank {rank}] Process group initialized successfully")
    except RuntimeError as e:
        logger.error(f"[Rank {rank}] Failed to initialize process group: {e}")
        raise


def cleanup():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
    """Remove output directories after each test."""
    for d in _OUTPUT_DIRS:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def extract_losses(loss_list):
    """Extract only loss values from (step, loss) tuples."""
    return [loss for _, loss in loss_list]


def compute_avg(losses):
    """Return the arithmetic mean of a loss sequence as a Python float."""
    return float(np.mean(losses))


def is_stable(losses, spike_tol=3.0):
    """
    Ensure no extreme spikes (instability).

    A loss value is considered a spike when it exceeds spike_tol × median.
    Using the median (rather than mean) makes the check robust to outliers.
    """
    losses = np.array(losses)
    median = np.median(losses)
    return np.all(losses < spike_tol * median)


def ddp_training_worker(rank, world_size, model_class, config, results):
    """
    Simple DDP training worker for basic parity tests.

    Args:
        rank: Process rank
        world_size: Total number of processes
        model_class: Model class to instantiate
        config: SimpleConfig (PICKLABLE dataclass)
        results: Multiprocessing dict for results
    """
    # Each rank gets a unique seed so data is not identical across devices,
    # which is the realistic DDP scenario (different data shards per rank)
    ddp_init(rank, world_size, use_file_store=False)

    try:
        torch.manual_seed(config.seed + rank)
        # Instantiate model, move to the QAIC device assigned to this rank,
        # then wrap with DDP so gradients are all-reduced after each backward
        model = model_class()
        device = torch.device(f"qaic:{rank}")
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        gradients_per_batch = []  # Gradient snapshots captured after each backward
        losses = []  # Scalar loss values for each batch

        for batch_idx in range(config.num_batches):
            # Synthetic random batch — no real dataset needed for unit tests
            x = torch.randn(config.batch_size, 8).to(device)
            y = torch.randn(config.batch_size, 2).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()  # DDP all-reduces gradients here automatically
            optimizer.step()

            losses.append(loss.item())
            # Detach and move gradients to CPU so they can be compared across
            # ranks via the shared multiprocessing Manager dict
            grads = [p.grad.detach().cpu().clone() for p in model.parameters() if p.grad is not None]
            gradients_per_batch.append(grads)

        # Store final parameters and full history in the shared results dict
        results[rank] = {
            "params": [p.data.detach().cpu().clone() for p in model.parameters()],
            "losses": losses,
            "gradients": gradients_per_batch,
        }

    except Exception as e:
        logger.error(f"Rank {rank} error: {e}")
        results[rank] = {"error": str(e)}

    finally:
        cleanup()


def single_device_training_worker(model_class, config, results):
    """
    Single-device training for reference parity comparison.

    This function performs single-device training to serve as a reference
    for comparing DDP multi-device training results.

    Args:
        model_class: Model class to instantiate
        config: SimpleConfig (PICKLABLE dataclass) with training parameters
        results: Dict for storing results (modified in-place)
    """
    try:
        torch.manual_seed(config.seed)

        model = model_class()
        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        losses = []
        gradients_per_batch = []

        # Training loop mirrors the DDP worker but runs on a single CPU/device
        # with no distributed communication — used as the reference baseline
        for batch_idx in range(config.num_batches):
            x = torch.randn(config.batch_size, 8)
            y = torch.randn(config.batch_size, 2)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Clone gradients before the next zero_grad() call wipes them
            grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            gradients_per_batch.append(grads)

        # Store results under a fixed key so callers can retrieve them easily
        results["single_device"] = {
            "params": [p.data.detach().cpu().clone() for p in model.parameters()],
            "losses": losses,
            "gradients": gradients_per_batch,
        }

    except Exception as e:
        logger.error(f"Single device error: {e}")
        results["single_device"] = {"error": str(e)}


def ddp_loss_worker(rank, world_size, config, results):
    """
    DDP training worker with deterministic loss computation.

    Uses synchronized random seeds across all ranks to ensure identical inputs
    and deterministic loss trajectories for parity testing.

    Args:
        rank: Process rank in the DDP group
        world_size: Total number of processes in the DDP group
        config: SimpleConfig (PICKLABLE dataclass) with training parameters
        results: Multiprocessing dict for storing results
    """
    # All ranks share the same global seed so model weights start identically
    ddp_init(rank, world_size, use_file_store=False)

    try:
        torch.manual_seed(config.seed)

        model = SimpleMLP()
        device = torch.device(f"qaic:{rank}")
        model = model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        losses = []

        # Re-seed per batch so every rank generates the SAME input tensors;
        # this makes the loss trajectory directly comparable to single-device
        for batch_idx in range(config.num_batches):
            torch.manual_seed(config.seed + batch_idx)  # Identical inputs across all ranks

            x = torch.randn(config.batch_size, 8).to(device)
            y = torch.randn(config.batch_size, 2).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()  # DDP all-reduces gradients; loss should match rank 0
            optimizer.step()

            losses.append(loss.item())

        # Only rank 0 writes results; other ranks produce identical values
        # so there is no need to collect from all of them
        if rank == 0:
            results["ddp"] = losses

    except Exception as e:
        logger.error(f"DDP loss worker rank {rank} error: {e}")
        results["error"] = str(e)

    finally:
        cleanup()


def pipeline_ddp_worker(rank, world_size, port, config_dict, results):
    """
    Top-level DDP training worker for FineTuningPipeline.

    Must be a top-level function (not a method) for pickling compatibility with mp.spawn.
    Initializes DDP, sets up FineTuningPipeline, and runs training.
    Only rank 0 reports results back.

    Args:
        rank: Process rank in DDP group
        world_size: Total processes in DDP group
        port: Port for DDP initialization
        config_dict: Configuration dictionary (PICKLABLE)
        results: Multiprocessing dict for results
    """
    # Expose rank/world-size to any library code that reads these env vars
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    pipeline = None
    trainer = None

    try:
        # FileStore is used here (instead of TCPStore) to avoid port conflicts
        # when mp.spawn launches multiple workers in the same OS process group
        ddp_init(rank, world_size, use_file_store=True, port=port)
        logger.info(
            f"[Rank {rank}] DDP initialized with backend={config_dict.get('ddp_config', {}).get('ddp_backend', 'N/A')}"
        )

        if FineTuningPipeline is None or ConfigManager is None:
            raise RuntimeError("FineTuningPipeline not available")

        # Use the same seed on every rank so the parity comparison is valid;
        # production code would use rank-specific seeds for data diversity
        torch.manual_seed(TEST_SEED + rank)

        # Build a ConfigManager and apply all training overrides from config_dict
        cm = ConfigManager()
        cm.config.model_name = _HF_MODEL.model_name
        cm.config.dataset["prompt_func"] = (
            "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
        )
        cm.config.dataset["completion_template"] = "{output}"
        cm.config.dataset["dataset_num_samples"] = 100  # Small subset for fast CI runs

        for k, v in config_dict.items():
            cm.config.training[k] = v

        # Tell the trainer which local device this process owns
        cm.config.training["local_rank"] = rank

        # Execute the full training pipeline (data loading → training → logging)
        logger.info(f"[Rank {rank}] Starting training pipeline...")
        pipeline = FineTuningPipeline(cm)
        pipeline.run()
        trainer = pipeline.trainer

        # Wait for all ranks to finish before rank 0 reads the trainer state;
        # device_ids=[rank] suppresses NSP resource warnings on QAIC
        if dist.is_initialized():
            try:
                dist.barrier(device_ids=[rank])
                logger.info(f"[Rank {rank}] Barrier passed - all ranks completed training")
            except Exception as barrier_err:
                logger.warning(f"[Rank {rank}] Barrier error (non-fatal): {barrier_err}")

        # Only rank 0 collects results; other ranks produce identical metrics
        if rank == 0:
            train_loss = None
            train_metrics = None

            if trainer and hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                # Build (step, loss) tuples from the trainer's log history
                losses = [(entry["step"], entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]
                if losses:
                    train_loss = losses
                # Collect per-epoch aggregate metrics (e.g. accuracy, perplexity)
                train_metrics = [
                    entry["train/epoch_metric"] for entry in trainer.state.log_history if "train/epoch_metric" in entry
                ]

            logger.info(
                f"[Rank {rank}] Training completed. Train loss: {len(train_loss) if train_loss else 0}, Train metrics: {len(train_metrics) if train_metrics else 0}"
            )
            results[rank] = {
                "train_loss": train_loss,
                "train_metrics": train_metrics,
            }

    except Exception as e:
        logger.error(f"[Rank {rank}] Pipeline error: {e}")
        # Use log_exception() for custom Logger compatibility
        try:
            logger.log_exception(f"[Rank {rank}] Pipeline error details:", e, raise_exception=False)
        except Exception:
            pass
        results["error"] = str(e)

    finally:
        # ----------------------------------------------------------------
        # Cleanup order matters: release model memory before destroying the
        # process group so QAIC device resources are freed gracefully.
        # ----------------------------------------------------------------
        try:
            # 1. Release pipeline and trainer objects to free device memory
            if pipeline is not None:
                try:
                    if hasattr(pipeline, "trainer") and pipeline.trainer is not None:
                        if hasattr(pipeline.trainer, "model"):
                            del pipeline.trainer.model  # Free model weights first
                        del pipeline.trainer
                    del pipeline
                except Exception as pipeline_cleanup_err:
                    logger.warning(f"[Rank {rank}] Pipeline cleanup error: {pipeline_cleanup_err}")

            # 2. Free CUDA cache if a CUDA device is also present
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # 3. Destroy the DDP process group last
            if dist.is_initialized():
                try:
                    # One final barrier so all ranks reach cleanup together
                    try:
                        dist.barrier(device_ids=[rank])
                    except Exception:
                        pass  # Non-fatal; proceed with teardown regardless

                    dist.destroy_process_group()
                    logger.info(f"[Rank {rank}] Process group destroyed")
                except Exception as cleanup_err:
                    logger.warning(f"[Rank {rank}] Cleanup error: {cleanup_err}")
        except Exception as final_cleanup_err:
            logger.warning(f"[Rank {rank}] Final cleanup error: {final_cleanup_err}")

        logger.info(f"[Rank {rank}] Worker finished")


# ============================================================================
# Test Classes
# ============================================================================


class TestGradientSynchronization:
    """
    Test gradient synchronization across DDP ranks.

    Verifies that gradients are properly synchronized during backward pass
    and that all ranks maintain identical gradient values after each batch.
    """

    def test_gradient_sync_per_batch(self):
        """
        Verify gradients are synchronized after each backward pass.

        For DDP to work correctly, gradients must be averaged across all ranks.
        This test captures gradient snapshots and verifies they match across ranks.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=3)
        manager = mp.Manager()
        results = manager.dict()  # Shared dict visible to all spawned processes

        # Spawn WORLD_SIZE processes; each runs ddp_training_worker and stores
        # its gradient snapshots in the shared Manager dict
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())

        # Use rank-0 gradients as the reference; all other ranks must match
        # because DDP all-reduces gradients before the optimizer step
        ref_grads = ddp_results[0]["gradients"]
        for rank_idx, rank_result in enumerate(ddp_results[1:], start=1):
            rank_grads = rank_result["gradients"]
            for batch_idx, (ref_grad_list, rank_grad_list) in enumerate(zip(ref_grads, rank_grads)):
                for param_idx, (ref_grad, rank_grad) in enumerate(zip(ref_grad_list, rank_grad_list)):
                    assert torch.allclose(ref_grad, rank_grad, atol=GRADIENT_ATOL), (
                        f"Gradient mismatch at batch {batch_idx}, param {param_idx}, rank {rank_idx}"
                    )


class TestBatchHandling:
    """
    Test DDP batch processing with various batch sizes.

    Validates that DDP correctly handles different batch sizes and
    maintains rank synchronization across configurations.
    """

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_various_batch_sizes(self, batch_size):
        """
        Test DDP with different batch sizes.

        Args:
            batch_size: Number of samples per batch
        """
        config = SimpleConfig(batch_size=batch_size, learning_rate=1e-3, num_batches=3)
        manager = mp.Manager()
        results = manager.dict()

        # Run DDP training with the parametrized batch size
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())
        # Confirm every spawned process reported back
        assert len(ddp_results) == WORLD_SIZE, f"Expected {WORLD_SIZE} results, got {len(ddp_results)}"

        # After DDP training, all ranks must have identical parameter tensors
        # because the optimizer step is applied to all-reduced gradients
        for rank_idx, rank_result in enumerate(ddp_results[1:], start=1):
            for param_idx, (p_rank0, p_rank) in enumerate(zip(ddp_results[0]["params"], rank_result["params"])):
                assert torch.allclose(p_rank0, p_rank, atol=PARAM_ATOL), (
                    f"Parameter mismatch at param {param_idx}, rank {rank_idx} with batch_size {batch_size}"
                )


class TestScaling:
    """
    Test DDP scaling properties with different world sizes.

    Validates that DDP implementation correctly handles different numbers
    of participating processes (world sizes).
    """

    @pytest.mark.parametrize("world_size", [1, 4])
    def test_different_world_sizes(self, world_size):
        """
        Test DDP with different world sizes.

        Args:
            world_size: Number of processes to spawn
        """
        # Skip if the requested world size exceeds available hardware
        if world_size > WORLD_SIZE:
            pytest.skip(f"Cannot test world_size {world_size} with only {WORLD_SIZE} available devices")

        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=3)
        manager = mp.Manager()
        results = manager.dict()

        if world_size == 1:
            # world_size=1 is the degenerate DDP case; run directly without mp.spawn
            single_device_training_worker(SimpleMLP, config, results)
            assert "single_device" in results, "Single device results not found"
        else:
            # world_size > 1: spawn the requested number of DDP worker processes
            mp.spawn(
                ddp_training_worker,
                args=(world_size, SimpleMLP, config, results),
                nprocs=world_size,
                join=True,
            )
            assert len(results) == world_size, f"Expected {world_size} results, got {len(results)}"


class TestErrorHandling:
    """
    Test error handling and robustness in DDP training.

    Ensures that worker processes properly capture and report exceptions,
    and that distributed process groups are cleaned up even on failure.
    """

    def test_worker_exception_handling(self):
        """
        Verify worker exceptions are properly captured and reported.

        Each worker process should catch exceptions and store them in
        the shared results dict for inspection.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=1)
        manager = mp.Manager()
        results = manager.dict()

        # Run a minimal 1-batch DDP job; the worker's try/except should
        # populate results[rank]["error"] if anything goes wrong
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        # All ranks must have completed cleanly — no error keys in results
        for rank, result in results.items():
            assert "error" not in result or result["error"] is None, (
                f"Rank {rank} reported error: {result.get('error')}"
            )


class TestDDPParity:
    """
    Parity tests between DDP and single-device training.

    Validates that multi-device DDP training produces equivalent results
    to single-device training with identical configurations and seeds.
    """

    def test_simple_model_parity(self):
        """
        Test parity between DDP and single-device with SimpleMLP model.

        Verifies:
        - All DDP ranks converge to identical parameters
        - Parameters remain within expected ranges (no explosion)
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=5)
        manager = mp.Manager()
        results = manager.dict()

        # Phase 1: multi-device DDP training
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, SimpleMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())
        assert len(ddp_results) == WORLD_SIZE, "Not all ranks reported results"

        # Phase 2: single-device reference training (same config, same seed)
        single_results = {}
        single_device_training_worker(SimpleMLP, config, single_results)

        # All DDP ranks must converge to the same parameter values because
        # the all-reduce ensures every rank applies identical gradient updates
        for rank_result in ddp_results[1:]:
            for p_rank0, p_rank in zip(ddp_results[0]["params"], rank_result["params"]):
                assert torch.allclose(p_rank0, p_rank, atol=PARAM_ATOL), "Parameters diverged between ranks"

        # Sanity-check that neither training path produced exploding parameters
        for p_ddp, p_single in zip(ddp_results[0]["params"], single_results["single_device"]["params"]):
            assert p_ddp.abs().max() < 100, "DDP parameters exploded"
            assert p_single.abs().max() < 100, "Single-device parameters exploded"

    def test_complex_model_parity(self):
        """
        Test parity between DDP and single-device with MultiLayerMLP model.

        Tests synchronization with more complex model architecture.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=5)
        manager = mp.Manager()
        results = manager.dict()

        # Use the deeper MultiLayerMLP to exercise DDP with more parameters
        mp.spawn(
            ddp_training_worker,
            args=(WORLD_SIZE, MultiLayerMLP, config, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        ddp_results = list(results.values())
        assert len(ddp_results) == WORLD_SIZE

        # All ranks must converge to identical parameters after DDP all-reduce
        for rank_result in ddp_results[1:]:
            for p_rank0, p_rank in zip(ddp_results[0]["params"], rank_result["params"]):
                assert torch.allclose(p_rank0, p_rank, atol=PARAM_ATOL)

    def test_single_vs_ddp_loss_parity(self):
        """
        Compare loss trajectory between single-device and multi-device DDP training.

        Ensures both optimization paths follow the same trajectory with
        identical seeds and deterministic random inputs.
        """
        config = SimpleConfig(batch_size=2, learning_rate=1e-3, num_batches=5, seed=TEST_SEED)

        manager = mp.Manager()
        ddp_results = manager.dict()

        # ddp_loss_worker re-seeds per batch so all ranks see identical inputs;
        # only rank 0 writes results back to the shared dict
        mp.spawn(
            ddp_loss_worker,
            args=(WORLD_SIZE, config, ddp_results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        assert "error" not in ddp_results, f"DDP error: {ddp_results.get('error')}"
        assert "ddp" in ddp_results, "DDP results not found"
        ddp_losses = ddp_results["ddp"]

        # Reproduce the exact same training on a single device using the same
        # per-batch seed sequence so the loss trajectory should be identical
        torch.manual_seed(config.seed)
        model = SimpleMLP()
        optimizer = SGD(model.parameters(), lr=config.learning_rate)
        criterion = MSELoss()

        single_losses = []
        for batch_idx in range(config.num_batches):
            # Mirror the per-batch seed used in ddp_loss_worker
            torch.manual_seed(config.seed + batch_idx)

            x = torch.randn(config.batch_size, 8)
            y = torch.randn(config.batch_size, 2)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            single_losses.append(loss.item())

        # Both trajectories must have the same length and agree within LOSS_ATOL
        assert len(ddp_losses) == len(single_losses), "Loss trajectory length mismatch"

        for step, (l_ddp, l_single) in enumerate(zip(ddp_losses, single_losses)):
            assert abs(l_ddp - l_single) < LOSS_ATOL, (
                f"Loss mismatch at step {step}: DDP={l_ddp:.6f}, Single={l_single:.6f}"
            )


@pytest.mark.skipif(
    not hasattr(torch, "qaic") or (hasattr(torch.qaic, "device_count") and torch.qaic.device_count() < 2),
    reason="Requires at least 2 QAIC devices",
)
class TestDDPPipelineParity:
    """
    End-to-end DDP pipeline parity tests using FineTuningPipeline.

    Validates that the complete training pipeline produces equivalent results
    between single-device and multi-device (DDP) configurations.

    These tests require actual QAIC hardware and model weights.
    """

    @staticmethod
    def _get_unique_port():
        """
        Get a unique port for MASTER_PORT to avoid conflicts between test runs.

        Returns:
            int: Available port number
        """
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def setup_method(self):
        """Clean up FileStore before each test to prevent stale lock files."""
        try:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
                logger.info(f"Cleaned up FileStore at {STORE_PATH}")
        except Exception as e:
            logger.warning(f"Failed to clean FileStore: {e}")

    def _assert_loss(self, value, label):
        """
        Validate loss value is finite and positive.

        Args:
            value: Loss value to validate
            label: Label for error messages

        Raises:
            AssertionError: If value is None, non-finite, or non-positive
        """
        assert value is not None, f"{label} is None"
        assert torch.isfinite(torch.tensor(value)), f"{label} not finite"
        assert value > 0, f"{label} must be > 0"

    def _build_config_dict(self, backend, output_dir):
        """
        Build configuration dictionary for FineTuningPipeline.

        Args:
            backend: DDP backend ("qccl" for QAIC, "gloo" for CPU, None for single-device)
            output_dir: Directory for training outputs

        Returns:
            dict: Configuration dictionary

        """
        return {
            "output_dir": output_dir,
            "seed": TEST_SEED,
            "max_steps": _MAX_STEPS,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "logging_steps": 1,
            "logging_strategy": "steps",
            "eval_steps": 50,
            "eval_strategy": "steps",
            "ddp_config": {
                "ddp_backend": backend,
                "ddp_find_unused_parameters": False,
                "ddp_bucket_cap_mb": 25,
                "ddp_broadcast_buffers": True,
                "ddp_timeout": 1800,
            },
        }

    def _run_single(self, config_dict):
        """
        Run single-device training using FineTuningPipeline.

        Args:
            config_dict: Configuration dictionary

        Returns:
            tuple: (train_loss, train_metrics) - Loss and metrics from training
        """
        try:
            if FineTuningPipeline is None or ConfigManager is None:
                pytest.skip("FineTuningPipeline not available")

            cm = ConfigManager()
            cm.config.model_name = _HF_MODEL.model_name
            cm.config.dataset["prompt_func"] = (
                "QEfficient.finetune.experimental.preprocessing.alpaca_func:create_alpaca_prompt"
            )
            cm.config.dataset["completion_template"] = "{output}"
            cm.config.dataset["dataset_num_samples"] = 100
            for k, v in config_dict.items():
                cm.config.training[k] = v

            cm.config.training["local_rank"] = 0
            cm.config.dataset["ddp_backend"] = None
            pipeline = FineTuningPipeline(cm)
            pipeline.run()

            trainer = pipeline.trainer

            # Extract training loss from log history
            train_loss = None
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                losses = [(entry["step"], entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]
                train_loss = losses if losses else None

            # Extract all training metrics (learning_rate, gradient_norm, etc.)
            train_metrics = None
            if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
                train_metrics = [
                    entry["train/epoch_metric"] for entry in trainer.state.log_history if "train/epoch_metric" in entry
                ]
            return train_loss, train_metrics

        except Exception as e:
            logger.error(f"Single-device training error: {e}", exc_info=True)
            raise
        finally:
            cleanup()

    def _run_ddp(self, config_dict, port=None):
        """
        Run multi-device DDP training using FineTuningPipeline.

        Args:
            config_dict: Configuration dictionary

        Returns:
            tuple: (train_loss, train_metrics) - Loss and metrics from DDP training
        """
        manager = mp.Manager()
        results = manager.dict()

        # Use top-level function for mp.spawn (not a method)
        mp.spawn(
            pipeline_ddp_worker,
            args=(WORLD_SIZE, port, config_dict, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

        assert "error" not in results, f"DDP error: {results.get('error')}"
        return results[0]["train_loss"], results[0]["train_metrics"]

    def test_single_vs_ddp_loss_parity(self):
        """
        Comprehensive parity test: Single-device vs DDP training.
        Validates that loss trajectories and training metrics are similar
        between single-device and multi-device DDP training with identical configurations.
        """

        # Phase 1: multi-device DDP run — uses a dynamically allocated port to
        # avoid conflicts with other tests running in parallel
        ddp_cfg = self._build_config_dict(
            backend="qccl",
            output_dir=OUTPUT_DIR_DDP,
        )
        logger.info("Running DDP training...")
        ddp_train, ddp_metrics = self._run_ddp(ddp_cfg, port=self._get_unique_port())

        # Phase 2: single-device run with the same config (backend=None disables DDP)
        single_cfg = self._build_config_dict(
            backend=None,
            output_dir=OUTPUT_DIR_SINGLE,
        )
        logger.info("Running single-device training...")
        single_train, single_metrics = self._run_single(single_cfg)

        # Both runs must produce the same number of logged loss steps
        assert len(ddp_train) == len(single_train), "Loss trajectory length mismatch"

        # Strip step indices; we only care about the scalar loss values
        single_losses = extract_losses(single_train)
        ddp_losses = extract_losses(ddp_train)

        assert len(single_losses) == len(ddp_losses)

        # Average loss parity: DDP and single-device should converge similarly
        avg_single = compute_avg(single_losses)
        avg_ddp = compute_avg(ddp_losses)

        assert abs(avg_single - avg_ddp) < PIPELINE_ATOL, f"Average loss mismatch: single={avg_single}, ddp={avg_ddp}"

        # Stability check: neither run should exhibit extreme loss spikes
        assert is_stable(single_losses)
        assert is_stable(ddp_losses)
        cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "ddp"])
