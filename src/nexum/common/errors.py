class NexumError(Exception):
    prefix = "[Nexum]"

    def __init__(self, message: str):
        error_name = self.__class__.__name__.replace("Nexum", "")
        super().__init__(f"{self.prefix} {error_name}: {message}")

    def __str__(self):
        return self.args[0]


class NexumRuntimeError(NexumError): ...


class NexumValueError(NexumError): ...
