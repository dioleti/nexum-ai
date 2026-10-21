import importlib
import json
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from nexum_ai_common import nexum_logging


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.root_logger = logging.getLogger()
        self.original_handlers = list(self.root_logger.handlers)
        self.original_level = self.root_logger.level

    def tearDown(self):
        self.root_logger.handlers = self.original_handlers
        self.root_logger.setLevel(self.original_level)

    def test_local_formatter(self):
        formatter = nexum_logging.NexumLocalFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )
        formatted = formatter.format(record)
        self.assertIn("[Nexum]", formatted)
        self.assertIn("INFO", formatted)
        self.assertIn("Hello world", formatted)
        self.assertIn(nexum_logging.COLORS["INFO"], formatted)

    def test_datadog_formatter_without_tracer(self):
        formatter = nexum_logging.NexumDatadogFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warn msg",
            args=(),
            exc_info=None,
        )
        with patch.object(nexum_logging, "dd_tracer", None):
            output = formatter.format(record)
            data = json.loads(output)
            self.assertEqual(data["prefix"], "Nexum")
            self.assertEqual(data["level"], "WARNING")
            self.assertEqual(data["message"], "Warn msg")
            self.assertEqual(data["logger"], "test_logger")
            self.assertNotIn("dd.trace_id", data)

    def test_datadog_formatter_with_tracer_and_span(self):
        formatter = nexum_logging.NexumDatadogFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error msg",
            args=(),
            exc_info=None,
        )
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span.trace_id = 12345
        mock_span.span_id = 67890
        mock_tracer.current_span.return_value = mock_span

        with patch.object(nexum_logging, "dd_tracer", mock_tracer):
            output = formatter.format(record)
            data = json.loads(output)
            self.assertEqual(data["dd.trace_id"], 12345)
            self.assertEqual(data["dd.span_id"], 67890)

    def test_elastic_formatter_without_apm(self):
        formatter = nexum_logging.NexumElasticFormatter()
        record = logging.LogRecord(
            name="elastic_test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Debug msg",
            args=(),
            exc_info=None,
        )
        with patch.object(nexum_logging, "elasticapm", None):
            output = formatter.format(record)
            data = json.loads(output)
            self.assertEqual(data["service"], "nexum")
            self.assertEqual(data["log.level"], "DEBUG")
            self.assertEqual(data["message"], "Debug msg")
            self.assertEqual(data["logger"], "elastic_test")
            self.assertNotIn("trace.id", data)

    def test_elastic_formatter_with_apm(self):
        formatter = nexum_logging.NexumElasticFormatter()
        record = logging.LogRecord(
            name="elastic_test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Info msg",
            args=(),
            exc_info=None,
        )
        mock_ctx = MagicMock()
        mock_ctx.get_trace_id.return_value = "trace-abc"
        mock_ctx.get_transaction_id.return_value = "tx-123"

        with patch.object(nexum_logging, "elasticapm", MagicMock()), patch.object(
            nexum_logging, "execution_context", mock_ctx
        ):
            output = formatter.format(record)
            data = json.loads(output)
            self.assertEqual(data["trace.id"], "trace-abc")
            self.assertEqual(data["transaction.id"], "tx-123")

    @patch("nexum_ai_common.nexum_logging.load_dotenv")
    def test_setup_logging_modes(self, mock_dotenv):
        with patch.dict(os.environ, {"NEXUM_LOG_MODE": "LOCAL"}, clear=True):
            nexum_logging.setup_logging()
            self.assertEqual(self.root_logger.level, logging.DEBUG)
            self.assertIsInstance(
                self.root_logger.handlers[-1].formatter,
                nexum_logging.NexumLocalFormatter,
            )

        with patch.dict(
            os.environ,
            {"NEXUM_LOG_MODE": "DATADOG", "LOG_LEVEL": "warning"},
            clear=True,
        ):
            nexum_logging.setup_logging()
            self.assertEqual(self.root_logger.level, logging.WARNING)
            self.assertIsInstance(
                self.root_logger.handlers[-1].formatter,
                nexum_logging.NexumDatadogFormatter,
            )

        with patch.dict(os.environ, {"NEXUM_LOG_MODE": "ELASTIC"}, clear=True):
            nexum_logging.setup_logging()
            self.assertEqual(self.root_logger.level, logging.INFO)
            self.assertIsInstance(
                self.root_logger.handlers[-1].formatter,
                nexum_logging.NexumElasticFormatter,
            )

        with patch.dict(os.environ, {"NEXUM_LOG_MODE": "UNKNOWN"}, clear=True):
            nexum_logging.setup_logging()
            self.assertEqual(self.root_logger.level, logging.INFO)
            self.assertIsInstance(
                self.root_logger.handlers[-1].formatter,
                nexum_logging.NexumLocalFormatter,
            )

    def test_module_import_with_mocked_ddtrace_and_elastic(self):
        mock_ddtrace = MagicMock()
        mock_elastic = MagicMock()
        mock_elastic_traces = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "ddtrace": mock_ddtrace,
                "elasticapm": mock_elastic,
                "elasticapm.traces": mock_elastic_traces,
            },
        ):
            reloaded = importlib.reload(nexum_logging)
            self.assertIsNotNone(reloaded.dd_tracer)
            self.assertIsNotNone(reloaded.elasticapm)


if __name__ == "__main__":
    unittest.main()
