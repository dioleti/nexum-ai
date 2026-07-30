import logging

for name in logging.root.manager.loggerDict:
    if name.startswith("ddtrace"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
