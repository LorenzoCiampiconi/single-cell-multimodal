from __future__ import annotations

import functools
import pathlib
import typing

from single_cell_multimodal_core import settings


if typing.TYPE_CHECKING:
    from typing import Callable  # pragma: no cover


def sanitize(fn: Callable[[str], pathlib.Path]):
    @functools.wraps(fn)
    def _wrapped_fn(*args, **kwargs):
        data_dir = fn(*args, **kwargs)
        if "." in data_dir.name:
            data_dir_to_make = data_dir.parent
        else:
            data_dir_to_make = data_dir
        data_dir_to_make.mkdir(parents=True, exist_ok=True)
        return data_dir.expanduser()

    return _wrapped_fn


@sanitize
def app_static_dir(identifier: str) -> pathlib.Path:
    p = getattr(settings.LOCATIONS, identifier)
    return pathlib.Path(p)


@sanitize
def resolve_data_path(label, file=None) -> pathlib.Path:
    if file is not None:
        return app_static_dir("DATA") / (
            f"{settings.DATA_PATHS[label].FOLDER}/{settings.DATA_PATHS[label].files[file]}"
        )
    else:
        return app_static_dir("DATA") / settings.DATA_PATHS[label].FOLDER
