from __future__ import annotations

import colorlog
import copy
import logging
import sys
import textwrap
import typing

from logging import handlers

from single_cell_multimodal_core import settings

if typing.TYPE_CHECKING:
    from typing import Optional  # pragma: no cover


logger = logging.getLogger(__name__)


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
    level = level if level is not None else settings.LOG.LEVEL
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level)
    handler = WrappingLinesStreamHandler(max_width=settings.LOG.COLUMN_WIDTH, stream=sys.stdout)
    handler.setFormatter(colorlog.ColoredFormatter(settings.LOG.FORMAT, datefmt=settings.LOG.DATE_FORMAT, style="%"))

    fileHandler = handlers.RotatingFileHandler(settings.LOG.log_path_to_file, maxBytes=20000000, backupCount=5)
    fileHandler.setFormatter(
        colorlog.ColoredFormatter(settings.LOG.FORMAT, datefmt=settings.LOG.DATE_FORMAT, style="%")
    )
    logger.addHandler(fileHandler)

    logger.addHandler(handler)


def get_banner() -> str:
    MAIN_DIR = SettingsContainer.MAIN_DIR
    banner = MAIN_DIR / "banner.txt"
    return banner.read_text()


def show_banner() -> None:
    banner = get_banner()
    for line in banner.splitlines():
        logger.info(line)


if __name__ == "__main__":
    setup_logging()
