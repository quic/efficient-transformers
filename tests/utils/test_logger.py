# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import time

import pytest

from QEfficient.utils.logging_utils import QEFFLogger


@pytest.fixture(autouse=True)
def reset_logger_state():
    QEFFLogger.close_logger()
    yield
    QEFFLogger.close_logger()
    # Keep process-global logger initialized for tests importing module-level adapters.
    QEFFLogger.get_logger("INFRA")


def test_logger_writes_json_records(tmp_path):
    logger = QEFFLogger.get_logger("model", "DEBUG", str(tmp_path))
    logger.info("hello logger")
    logger.warning("warning logger")

    log_path = QEFFLogger.get_logfile_path()
    assert log_path is not None

    QEFFLogger.close_logger()
    with open(log_path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    assert len(rows) >= 2
    assert rows[-1]["namespace"] == "model"
    assert rows[-1]["level"] == "WARNING"
    assert rows[-1]["message"] == "warning logger"
    assert isinstance(rows[-1]["created"], float)


def test_print_table_from_logged_milestones(tmp_path, capsys):
    logger = QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    logger.info("Initiating the model weight loading.")
    time.sleep(0.01)
    logger.info("Pytorch transforms applied to model: test")
    time.sleep(0.01)
    logger.info("Model export is finished and saved at: /tmp/model.onnx")
    time.sleep(0.01)
    logger.info("Model compilation is finished and saved at: /tmp/model.qpc")
    time.sleep(0.01)
    logger.info("Text generation finished")

    assert QEFFLogger.print_table() is True
    output = capsys.readouterr().out
    assert "Model Loading" in output
    assert "Model Exporting" in output
    assert "Model Compilation" in output
    assert "Text Generation" in output
    assert "Total Time" in output


def test_print_table_without_log_file_returns_false():
    QEFFLogger.close_logger()
    assert QEFFLogger.print_table() is False
