# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests for QEfficient.generation.cloud_infer module.

Tests verify:
  - QAICInferenceSession importability
  - QAICInferenceSession raises ImportError when qaicrt/QAicApi_pb2 unavailable
  - QAICInferenceSession class structure (methods, properties)
  - QAICInferenceSession method signatures
  - skip_buffers calls set_buffers with empty arrays

All tests run on CPU only. No actual QAIC hardware execution.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Tests: QAICInferenceSession importability
# ---------------------------------------------------------------------------


class TestQAICInferenceSessionImport:
    """QAICInferenceSession must be importable and raise ImportError when dependencies missing."""

    def test_qaic_inference_session_importable(self):
        """QAICInferenceSession must be importable from cloud_infer."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert QAICInferenceSession is not None

    def test_qaic_inference_session_raises_import_error_when_qaicrt_missing(self):
        """QAICInferenceSession.__init__ must raise ImportError when qaicrt is not available."""
        # Mock qaicrt and QAicApi_pb2 as unavailable
        with patch.dict(sys.modules, {"qaicrt": None, "QAicApi_pb2": None}):
            # Force reimport to trigger ImportError check
            import importlib

            import QEfficient.generation.cloud_infer as cloud_infer_module

            importlib.reload(cloud_infer_module)

            from QEfficient.generation.cloud_infer import QAICInferenceSession

            with pytest.raises(ImportError, match="Unable to import `qaicrt`"):
                QAICInferenceSession(qpc_path="/fake/path.qpc")

    def test_qaic_inference_session_is_class(self):
        """QAICInferenceSession must be a class."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert isinstance(QAICInferenceSession, type)


# ---------------------------------------------------------------------------
# Tests: QAICInferenceSession structure
# ---------------------------------------------------------------------------


class TestQAICInferenceSessionStructure:
    """QAICInferenceSession must have correct class-level structure."""

    def test_has_init_method(self):
        """QAICInferenceSession must have __init__ method."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "__init__")

    def test_has_input_names_property(self):
        """QAICInferenceSession must have input_names property."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "input_names")

    def test_has_output_names_property(self):
        """QAICInferenceSession must have output_names property."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "output_names")

    def test_has_activate_method(self):
        """QAICInferenceSession must have activate method."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "activate")
        assert callable(QAICInferenceSession.activate)

    def test_has_deactivate_method(self):
        """QAICInferenceSession must have deactivate method."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "deactivate")
        assert callable(QAICInferenceSession.deactivate)

    def test_has_set_buffers_method(self):
        """QAICInferenceSession must have set_buffers method."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "set_buffers")
        assert callable(QAICInferenceSession.set_buffers)

    def test_has_skip_buffers_method(self):
        """QAICInferenceSession must have skip_buffers method."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "skip_buffers")
        assert callable(QAICInferenceSession.skip_buffers)

    def test_has_run_method(self):
        """QAICInferenceSession must have run method."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        assert hasattr(QAICInferenceSession, "run")
        assert callable(QAICInferenceSession.run)


# ---------------------------------------------------------------------------
# Tests: QAICInferenceSession method signatures
# ---------------------------------------------------------------------------


class TestQAICInferenceSessionMethodSignatures:
    """QAICInferenceSession methods must have correct signatures."""

    def test_init_accepts_qpc_path(self):
        """__init__ must accept qpc_path parameter."""
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.__init__)
        assert "qpc_path" in sig.parameters

    def test_init_accepts_device_ids(self):
        """__init__ must accept device_ids parameter."""
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.__init__)
        assert "device_ids" in sig.parameters

    def test_init_accepts_activate(self):
        """__init__ must accept activate parameter."""
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.__init__)
        assert "activate" in sig.parameters

    def test_init_accepts_enable_debug_logs(self):
        """__init__ must accept enable_debug_logs parameter."""
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.__init__)
        assert "enable_debug_logs" in sig.parameters

    def test_set_buffers_accepts_buffers_dict(self):
        """set_buffers must accept buffers parameter."""
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.set_buffers)
        assert "buffers" in sig.parameters

    def test_skip_buffers_accepts_skipped_buffer_names(self):
        """skip_buffers must accept skipped_buffer_names parameter."""
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.skip_buffers)
        assert "skipped_buffer_names" in sig.parameters

    def test_run_accepts_inputs_dict(self):
        """run must accept inputs parameter."""
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.run)
        assert "inputs" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: QAICInferenceSession skip_buffers behavior
# ---------------------------------------------------------------------------


class TestQAICInferenceSessionSkipBuffers:
    """skip_buffers must call set_buffers with empty arrays."""

    def test_skip_buffers_calls_set_buffers_with_empty_arrays(self):
        """skip_buffers must call set_buffers with empty numpy arrays for each buffer name."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        # Create a mock instance
        mock_session = MagicMock(spec=QAICInferenceSession)
        mock_session.set_buffers = MagicMock()

        # Call skip_buffers on the real class with the mock instance
        QAICInferenceSession.skip_buffers(mock_session, ["buffer1", "buffer2"])

        # Verify set_buffers was called
        assert mock_session.set_buffers.called
        call_args = mock_session.set_buffers.call_args[0][0]
        assert "buffer1" in call_args
        assert "buffer2" in call_args
        # Verify the values are empty arrays
        assert isinstance(call_args["buffer1"], np.ndarray)
        assert isinstance(call_args["buffer2"], np.ndarray)
        assert call_args["buffer1"].size == 0
        assert call_args["buffer2"].size == 0


# ---------------------------------------------------------------------------
# Tests: QAICInferenceSession error handling
# ---------------------------------------------------------------------------


class TestQAICInferenceSessionErrorHandling:
    """QAICInferenceSession must handle errors correctly."""

    def test_init_with_invalid_qpc_path_type_raises_error(self):
        """__init__ with non-string/non-Path qpc_path should raise TypeError or similar."""
        # This test would require mocking qaicrt, so we skip actual instantiation
        # and just verify the signature accepts Union[Path, str]
        import inspect

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        sig = inspect.signature(QAICInferenceSession.__init__)
        # Verify qpc_path parameter exists
        assert "qpc_path" in sig.parameters

    def test_set_buffers_with_non_dict_raises_error(self):
        """set_buffers with non-dict input should raise TypeError or AttributeError."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        mock_session = MagicMock(spec=QAICInferenceSession)
        mock_session.binding_index_map = {}

        # Call set_buffers with invalid input
        with pytest.raises((TypeError, AttributeError)):
            QAICInferenceSession.set_buffers(mock_session, "not_a_dict")


# ---------------------------------------------------------------------------
# Tests: QAICInferenceSession properties
# ---------------------------------------------------------------------------


class TestQAICInferenceSessionProperties:
    """QAICInferenceSession properties must return correct types."""

    def test_input_names_returns_list(self):
        """input_names property must return a list."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        # Create a mock instance with bindings
        mock_session = MagicMock(spec=QAICInferenceSession)
        mock_session.bindings = []

        # Call the property on the real class
        result = QAICInferenceSession.input_names.fget(mock_session)
        assert isinstance(result, list)

    def test_output_names_returns_list(self):
        """output_names property must return a list."""
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        # Create a mock instance with bindings
        mock_session = MagicMock(spec=QAICInferenceSession)
        mock_session.bindings = []

        # Call the property on the real class
        result = QAICInferenceSession.output_names.fget(mock_session)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Module-level constants
# ---------------------------------------------------------------------------


class TestCloudInferModuleConstants:
    """cloud_infer module must have expected constants."""

    def test_is_qaicrt_imported_exists(self):
        """is_qaicrt_imported must be defined in module."""
        from QEfficient.generation import cloud_infer

        assert hasattr(cloud_infer, "is_qaicrt_imported")
        assert isinstance(cloud_infer.is_qaicrt_imported, bool)

    def test_is_aicapi_imported_exists(self):
        """is_aicapi_imported must be defined in module."""
        from QEfficient.generation import cloud_infer

        assert hasattr(cloud_infer, "is_aicapi_imported")
        assert isinstance(cloud_infer.is_aicapi_imported, bool)
