import logging
import os
import sys
import traceback
from loguru import logger

LOCAL_DEV = os.getenv("LOCAL_DEV")


def log_exception(*args):
    """[Captures the fullstack trace of uncaught exceptions, and provides them in a structured format]"""
    if len(args) == 1:
        e = args[0]
        etype, value, tb = type(e), e, e.__traceback__
    elif len(args) == 3:
        etype, value, tb = args
    else:
        logger.error(
            "Not able to log exception. Wrong number of arguments given. Should either receive 1 argument "
            "- an exception, or 3 arguments: exc type, exc value and traceback"
        )
        return

    tb_parsed = []
    for filename, lineno, func, text in traceback.extract_tb(tb):
        tb_parsed.append(
            {"filename": filename, "lineno": lineno, "func": func, "text": text}
        )

    logger.error(
        str(value),
        exception=traceback.format_exception_only(etype, value)[0].strip(),
        traceback=tb_parsed,
    )


class InterceptHandler(logging.Handler):
    """
        A class that allows loguru library to plug into the standard logging libarary.
        Provides a way for imported libaries to also use the loguru logger.
        Please refer to the loguru docs for more information
    Args:
        logging : the standard libary logging libary
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger():
    """
    Setup the logger and return it.
    """
    logger.remove()
    if LOCAL_DEV:
        logger.add(sys.stdout, serialize=False, level="DEBUG")
        logger.add("logs/debug.log", serialize=False, level="DEBUG")
        logger.add("logs/error.log", serialize=False, level="ERROR")
    else:
        logger.add(sys.stdout, serialize=True)
        logger.add("logs/debug.log", serialize=True, level="DEBUG")
        logger.add("logs/error.log", serialize=True, level="ERROR")

        # capture uncaught exceptions
        sys.excepthook = log_exception

    # set up loguru to work with the standard logging module
    logging.basicConfig(handlers=[InterceptHandler()], level=30)
