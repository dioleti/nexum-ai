import json
import logging
import unittest
from unittest.mock import patch, MagicMock

from nexum.common.logging import (
    NexumLocalFormatter,
    NexumDatadogFormatter,
    NexumElasticFormatter,
    setup_logging,
)


class TestNexumLocalFormatter(unittest.TestCase):
    def test_format(self):
        formatter = NexumLocalFormatter("%(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", None, None)
        output = formatter.format(record)
        self.assertIn("[Nexum]", output)
        self.assertIn("INFO", output)
        self.assertIn("msg", output)


class TestNexumDatadogFormatter(unittest.TestCase):
    @patch("nexum.common.logging.dd_tracer")
    def test_format_with_trace(self, mock_tracer):
        span = MagicMock()
        span.trace_id = 123
        span.span_id = 456
        mock_tracer.current_span.return_value = span

        formatter = NexumDatadogFormatter()
        record = logging.LogRecord("test", logging.ERROR, "", 0, "error occurred", None, None)
        output = formatter.format(record)
        data = json.loads(output)

        self.assertEqual(data["level"], "ERROR")
        self.assertEqual(data["message"], "error occurred")
        self.assertEqual(data["dd.trace_id"], 123)
        self.assertEqual(data["dd.span_id"], 456)

    @patch("nexum.common.logging.dd_tracer", None)
    def test_format_without_trace(self):
        formatter = NexumDatadogFormatter()
        record = logging.LogRecord("test", logging.WARNING, "", 0, "warn", None, None)
        output = formatter.format(record)
        data = json.loads(output)

        self.assertEqual(data["level"], "WARNING")
        self.assertEqual(data["message"], "warn")
        self.assertNotIn("dd.trace_id", data)


class TestNexumElasticFormatter(unittest.TestCase):
    @patch("nexum.common.logging.elasticapm")
    @patch("nexum.common.logging.execution_context")
    def test_format_with_trace(self, mock_exec, mock_elastic):
        mock_exec.get_trace_id.return_value = "abc"
        mock_exec.get_transaction_id.return_value = "xyz"

        formatter = NexumElasticFormatter()
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "debugging", None, None)
        output = formatter.format(record)
        data = json.loads(output)

        self.assertEqual(data["log.level"], "DEBUG")
        self.assertEqual(data["message"], "debugging")
        self.assertEqual(data["trace.id"], "abc")
        self.assertEqual(data["transaction.id"], "xyz")

    @patch("nexum.common.logging.elasticapm", None)
    def test_format_without_trace(self):
        formatter = NexumElasticFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "info", None, None)
        output = formatter.format(record)
        data = json.loads(output)

        self.assertEqual(data["log.level"], "INFO")
        self.assertEqual(data["message"], "info")
        self.assertNotIn("trace.id", data)


class TestSetupLogging(unittest.TestCase):
    @patch("nexum.common.logging.load_dotenv")
    @patch("nexum.common.logging.os.getenv")
    def test_local_mode(self, mock_getenv, mock_load):
        mock_getenv.side_effect = lambda key, default=None: "LOCAL" if key == "NEXUM_LOG_MODE" else None

        setup_logging()
        root = logging.getLogger()
        handler = root.handlers[-1]
        self.assertEqual(root.level, logging.DEBUG)
        self.assertIsInstance(handler.formatter, NexumLocalFormatter)

    @patch("nexum.common.logging.load_dotenv")
    @patch("nexum.common.logging.os.getenv")
    def test_datadog_mode(self, mock_getenv, mock_load):
        mock_getenv.side_effect = lambda key, default=None: "DATADOG" if key == "NEXUM_LOG_MODE" else None

        setup_logging()
        root = logging.getLogger()
        handler = root.handlers[-1]
        self.assertEqual(root.level, logging.INFO)
        self.assertIsInstance(handler.formatter, NexumDatadogFormatter)

    @patch("nexum.common.logging.load_dotenv")
    @patch("nexum.common.logging.os.getenv")
    def test_elastic_mode(self, mock_getenv, mock_load):
        mock_getenv.side_effect = lambda key, default=None: "ELASTIC" if key == "NEXUM_LOG_MODE" else None

        setup_logging()
        root = logging.getLogger()
        handler = root.handlers[-1]
        self.assertEqual(root.level, logging.INFO)
        self.assertIsInstance(handler.formatter, NexumElasticFormatter)


if __name__ == "__main__":
    unittest.main()
