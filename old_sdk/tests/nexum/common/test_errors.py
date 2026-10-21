import unittest

from nexum.common.errors import NexumError, NexumRuntimeError, NexumValueError


class TestErrors(unittest.TestCase):
    def test_nexum_error_message(self):
        err = NexumError("something went wrong")
        self.assertEqual(str(err), "[Nexum] Error: something went wrong")

    def test_nexum_runtime_error(self):
        err = NexumRuntimeError("runtime fail")
        self.assertIsInstance(err, NexumError)
        self.assertEqual(str(err), "[Nexum] RuntimeError: runtime fail")

    def test_nexum_value_error(self):
        err = NexumValueError("invalid value")
        self.assertIsInstance(err, NexumError)
        self.assertEqual(str(err), "[Nexum] ValueError: invalid value")


if __name__ == "__main__":
    unittest.main()
