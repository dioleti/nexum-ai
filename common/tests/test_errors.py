import unittest

from nexum_ai_common.errors import NexumError, NexumRuntimeError, NexumValueError


class TestErrors(unittest.TestCase):
    def test_nexum_error_message(self):
        err = NexumError("something went wrong")
        self.assertEqual(str(err), "[Nexum] Error: something went wrong")
        self.assertIsInstance(err, Exception)

    def test_nexum_runtime_error_message(self):
        err = NexumRuntimeError("runtime failure")
        self.assertEqual(str(err), "[Nexum] RuntimeError: runtime failure")
        self.assertIsInstance(err, NexumError)

    def test_nexum_value_error_message(self):
        err = NexumValueError("invalid value")
        self.assertEqual(str(err), "[Nexum] ValueError: invalid value")
        self.assertIsInstance(err, NexumError)


if __name__ == "__main__":
    unittest.main()
