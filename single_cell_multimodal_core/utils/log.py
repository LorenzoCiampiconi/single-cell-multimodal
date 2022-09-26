from __future__ import annotations

import colorlog
import copy
import logging
import sys
import textwrap
import typing

from logging import handlers


if typing.TYPE_CHECKING:
    from typing import Optional  # pragma: no cover


logger = logging.getLogger(__name__)

settings = {
    "log": {
        "log_path_to_file": "/Users/lciampiconi/PycharmProjects/kaggle/single-cell-multimodal/pipe.log",
        "level": "INFO",
        "column_width": 8000,
        "format": "%(thin_white)s%(asctime)s %(log_color)s%(levelname)-5s%(thin_cyan)s%(name)-30s %(bold_white)s%(message)s%(reset)s",
        "date_format": "%m-%d %H:%M:%S",
    }
}


class WrappingLinesStreamHandler(logging.StreamHandler):
    def __init__(self, *args, max_width: int = 8000, **kwargs):
        self.max_width = max_width
        self._wrapper = textwrap.TextWrapper(width=max_width, subsequent_indent="\t", break_long_words=False)
        super(WrappingLinesStreamHandler, self).__init__(*args, **kwargs)

    def emit(self, record) -> None:
        message = record.getMessage()
        for wrapped_line in self._wrapper.wrap(message):
            wrapped_record = copy.copy(record)
            wrapped_record.msg = wrapped_line
            wrapped_record.args = ()
            super(WrappingLinesStreamHandler, self).emit(wrapped_record)


def setup_logging(level: Optional[str] = None) -> None:
    level = level if level is not None else settings["log"]["level"]
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level)
    handler = WrappingLinesStreamHandler(max_width=settings["log"]["column_width"], stream=sys.stdout)
    handler.setFormatter(
        colorlog.ColoredFormatter(settings["log"]["format"], datefmt=settings["log"]["date_format"], style="%")
    )

    fileHandler = handlers.RotatingFileHandler(settings["log"]["log_path_to_file"], maxBytes=20000000, backupCount=5)
    fileHandler.setFormatter(
        colorlog.ColoredFormatter(settings["log"]["format"], datefmt=settings["log"]["date_format"], style="%")
    )
    logger.addHandler(fileHandler)

    logger.addHandler(handler)


if __name__ == "__main__":
    setup_logging()
