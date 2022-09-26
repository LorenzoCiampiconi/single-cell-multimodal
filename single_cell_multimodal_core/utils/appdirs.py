import functools
from pathlib import Path
import typing

from typing import Callable  # pragma: no cover


def sanitize(fn: Callable[[str], Path]):
    @functools.wraps(fn)
    def _wrapped_fn(*args, **kwargs):
        data_dir = fn(*args, **kwargs)
        if data_dir.is_file():
            data_dir_to_make = data_dir.parent
        else:
            data_dir_to_make = data_dir
        data_dir_to_make.mkdir(parents=True, exist_ok=True)
        return data_dir.expanduser()

    return _wrapped_fn


@sanitize
def app_static_dir(child: str) -> Path:
    return Path(__file__).parent.parent.parent.absolute() / child
