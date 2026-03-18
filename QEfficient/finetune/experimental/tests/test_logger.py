# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
from unittest.mock import patch

import pytest

from QEfficient.finetune.experimental.core.logger import Logger, get_logger


class TestLogger:
    def setup_method(self):
        """Reset the global logger before each test method"""
        import QEfficient.finetune.experimental.core.logger as logger_module

        logger_module._logger = None

    def test_init_console_only(self):
        """Test logger initialization with console-only output"""
        logger = Logger("test_logger")

        # Check logger attributes
        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.INFO

        # Check handlers - should have console handler only
        assert len(logger.logger.handlers) == 1  # Only console handler
        assert isinstance(logger.logger.handlers[0], logging.StreamHandler)

    def test_init_with_file(self, tmp_path):
        """Test logger initialization with file output"""
        log_file = tmp_path / "test.log"
        logger = Logger("file_test_logger", str(log_file))

        # Check handlers - should have both console and file handlers
        assert len(logger.logger.handlers) == 2  # Console + file handler
        assert isinstance(logger.logger.handlers[0], logging.StreamHandler)
        assert isinstance(logger.logger.handlers[1], logging.FileHandler)

        # Check file creation
        assert log_file.exists()

    def test_log_levels(self, caplog):
        """Test all log levels work correctly"""
        logger = Logger("level_test_logger", level=logging.DEBUG)

        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

            # Check all messages were logged
            assert "Debug message" in caplog.text
            assert "Info message" in caplog.text
            assert "Warning message" in caplog.text
            assert "Error message" in caplog.text
            assert "Critical message" in caplog.text

    @patch("QEfficient.finetune.experimental.core.logger.get_local_rank")
    def test_log_rank_zero_positive_case(self, mock_get_local_rank, caplog):
        """Test rank zero logging functionality"""
        mock_get_local_rank.return_value = 0
        logger = Logger("rank_test_logger")

        with caplog.at_level(logging.INFO):
            logger.log_rank_zero("Rank zero message")

            assert "Rank zero message" in caplog.text

    @patch("QEfficient.finetune.experimental.core.logger.get_local_rank")
    def test_log_rank_zero_negative_case(self, mock_get_local_rank, caplog):
        """Test to verify that only rank‑zero messages are logged"""
        mock_get_local_rank.return_value = 1
        logger = Logger("rank_test_logger")

        with caplog.at_level(logging.INFO):
            logger.log_rank_zero("Should not appear")

            assert "Should not appear" not in caplog.text

    def test_log_exception_raise(self, caplog):
        """Test exception logging with raising"""
        logger = Logger("exception_test_logger")

        with pytest.raises(ValueError), caplog.at_level(logging.ERROR):
            logger.log_exception("Custom error", ValueError("Test exception"), raise_exception=True)

        # The actual logged message is "Custom error: Test exception"
        # But the exception itself contains just "Test exception"
        assert "Custom error: Test exception" in caplog.text

    def test_log_exception_no_raise(self, caplog):
        """Test exception logging without raising"""
        logger = Logger("exception_test_logger")

        with caplog.at_level(logging.ERROR):
            logger.log_exception("Custom error", ValueError("Test exception"), raise_exception=False)

            # Check that the formatted message was logged
            assert "Custom error: Test exception" in caplog.text

    def test_prepare_for_logs(self, tmp_path):
        """Test preparing logger for training logs"""
        output_dir = tmp_path / "output"
        logger = Logger("prepare_test_logger")

        # Prepare for logs
        logger.prepare_for_logs(str(output_dir), log_level="DEBUG")

        # Check file handler was added
        file_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

        # Check file exists
        log_file = output_dir / "training.log"
        assert log_file.exists()

        # Check log level was updated
        assert logger.logger.level == logging.DEBUG

    def test_prepare_for_logs_no_file_handler(self):
        """Test preparing logger without saving to file"""
        logger = Logger("prepare_test_logger")

        # Prepare for logs without saving metrics
        logger.prepare_for_logs(log_level="INFO")

        # Check no file handler was added
        file_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0

    def test_prepare_for_logs_already_has_file_handler(self, tmp_path):
        """Test preparing logger when file handler already exists"""
        output_dir = tmp_path / "output"
        logger = Logger("prepare_test_logger")

        # Add a file handler manually first
        log_file = output_dir / "manual.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        logger.logger.addHandler(file_handler)

        # Prepare for logs again
        logger.prepare_for_logs(str(output_dir), log_level="INFO")

        # Should still have only one file handler
        file_handlers = [h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_get_logger_singleton(self):
        """Test that get_logger returns the same instance"""
        logger1 = get_logger()
        logger2 = get_logger()

        assert logger1 is logger2

    def test_get_logger_with_file(self, tmp_path):
        """Test get_logger with file parameter"""
        log_file = tmp_path / "get_logger_test.log"
        logger = get_logger(str(log_file))

        # Check that we have 2 handlers (console + file)
        assert len(logger.logger.handlers) == 2  # Console + file
        assert isinstance(logger.logger.handlers[1], logging.FileHandler)

        # Check file exists
        assert log_file.exists()


class TestLoggerIntegration:
    """Integration tests for logger functionality"""

    def setup_method(self):
        """Reset the global logger before each test method"""
        import QEfficient.finetune.experimental.core.logger as logger_module

        logger_module._logger = None

    def test_complete_workflow(self, tmp_path, caplog):
        """Test complete logger workflow"""
        # Setup
        log_file = tmp_path / "workflow.log"
        logger = Logger("workflow_test", str(log_file), logging.DEBUG)

        # Test all methods
        logger.debug("Debug test")
        logger.info("Info test")
        logger.warning("Warning test")
        logger.error("Error test")
        logger.critical("Critical test")

        # Test exception handling
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_exception("Caught exception", e, raise_exception=False)

        # Test rank zero logging
        with patch("QEfficient.finetune.experimental.core.logger.get_local_rank") as mock_rank:
            mock_rank.return_value = 0
            logger.log_rank_zero("Rank zero test")

        # Verify all messages were logged
        with caplog.at_level(logging.DEBUG):
            assert "Debug test" in caplog.text
            assert "Info test" in caplog.text
            assert "Warning test" in caplog.text
            assert "Error test" in caplog.text
            assert "Critical test" in caplog.text
            assert "Caught exception: Test exception" in caplog.text
            assert "Rank zero test" in caplog.text

            # Check file was written to
            assert log_file.exists()
            content = log_file.read_text()
            assert "Debug test" in content
            assert "Info test" in content
            assert "Warning test" in content
            assert "Error test" in content
            assert "Critical test" in content
            assert "Caught exception: Test exception" in content
            assert "Rank zero test" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
