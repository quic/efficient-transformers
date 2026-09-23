# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import time
from pathlib import Path

import pytest

from QEfficient.utils.logging_utils import (
    QEFFLogger,
    log_api_arguments,
    log_generate_call,
    log_pipeline_api,
)


@pytest.fixture(autouse=True)
def reset_logger_state():
    QEFFLogger.close_logger()
    yield
    QEFFLogger.close_logger()
    # Keep process-global logger initialized for tests importing module-level adapters.
    QEFFLogger.get_logger("INFRA")


def test_log_api_arguments_writes_structured_json(tmp_path):
    QEFFLogger.get_logger("test", "INFO", str(tmp_path))
    log_api_arguments("compile", "QEFFTestModel", {"token": "secret", "path": Path("/tmp/model")})

    log_path = QEFFLogger.get_logfile_path()
    assert log_path is not None
    QEFFLogger.close_logger()
    with open(log_path, "r", encoding="utf-8") as handle:
        record = json.loads(next(handle))

    assert record["event"] == "api_call"
    assert record["api"] == "compile"
    assert record["arguments"] == {"token": "<redacted>", "path": "/tmp/model"}


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


def test_log_generate_call_captures_arguments_and_result(tmp_path):
    class TestModel:
        @log_generate_call
        def generate(self, generation_len=4):
            return generation_len * 2

    QEFFLogger.get_logger("TestModel", "INFO", str(tmp_path))
    assert TestModel().generate() == 8

    log_path = QEFFLogger.get_logfile_path()
    assert log_path is not None
    QEFFLogger.close_logger()
    with open(log_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    api_record = next(record for record in records if record.get("api") == "generate")
    assert api_record["arguments"] == {"generation_len": 4}
    assert any(record["message"] == "Generation completed." for record in records)


def test_cached_stages_are_zero_in_timing_table(tmp_path):
    QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    QEFFLogger.start_run("cached-model")
    logger = QEFFLogger.get_logger("infra")
    logger.info("Applied PyTorch transforms to model: cached-model.")
    logger.info("ONNX export skipped (cached QPC).")
    logger.info("Compilation skipped (cached QPC).")
    log_api_arguments("generate", "CachedModel", {"max_new_tokens": 1})
    logger.info("Generation completed.")

    table = QEFFLogger._build_timing_table(QEFFLogger._current_run_id)
    assert table is not None
    assert "| Model Exporting   |      0.000 |" in table
    assert "| Model Compilation |      0.000 |" in table


def test_pipeline_api_arguments_and_nested_records_are_simplified(tmp_path):
    class TestPipeline:
        @classmethod
        @log_pipeline_api("from_pretrained", "Model loading completed.", "load_complete")
        def from_pretrained(cls, model_name):
            log_api_arguments("from_pretrained", "ChildModel", {"model_name": model_name})
            return cls()

        @log_pipeline_api("export", "ONNX export completed.", "export_complete")
        def export(self):
            log_api_arguments("export", "ChildModel", {"duplicate": True})

        @log_pipeline_api("compile", "Compilation completed.", "compile_complete")
        def compile(self):
            log_api_arguments("compile", "ChildModel", {"duplicate": True})

    QEFFLogger.get_logger("TestPipeline", "INFO", str(tmp_path))
    pipeline = TestPipeline.from_pretrained("test/model")
    pipeline.export()
    pipeline.compile()
    QEFFLogger.close_logger()

    log_path = QEFFLogger.get_logfile_path()
    assert log_path is None
    files = list(tmp_path.glob("*.log"))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines() if line.startswith("{")]
    assert [record["api"] for record in records if record.get("event") == "api_call"] == [
        "from_pretrained",
        "export",
        "compile",
    ]


def test_failure_and_cleanup_are_logged_before_final_table(tmp_path):
    QEFFLogger.get_logger("infra", "INFO", str(tmp_path))
    QEFFLogger.start_run("test-model")
    logger = QEFFLogger.get_logger("infra")
    logger.info("Applied PyTorch transforms to model: test-model.")
    logger.info("Compilation completed.")
    QEFFLogger.log_api_failure("generate", "TestModel", RuntimeError("test failure"))
    logger.info("Cleanup completed.")
    QEFFLogger.close_logger()

    files = list(tmp_path.glob("*.log"))
    contents = files[0].read_text()
    assert '"event": "api_failure"' in contents
    summary_index = contents.index("===== QEfficient Timing Summary:")
    cleanup_index = contents.index('"message": "Cleanup completed."')
    assert cleanup_index < summary_index
    assert contents.rstrip().endswith("| Total Time        |      0.000 |")
