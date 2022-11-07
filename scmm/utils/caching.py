import logging
from functools import wraps
from pathlib import Path
from typing import Iterable, Callable, Optional

import numpy as np

from scmm.utils.appdirs import app_static_dir

logger = logging.getLogger(__name__)


def np_load_wrapper(svd_caching_path):
    with np.load(svd_caching_path) as npz_file:
        cached_output = npz_file["cached_output"]
        return cached_output


def np_save_wrapper(cached_output, svd_caching_path):
    np.savez_compressed(svd_caching_path, cached_output=cached_output)


def caching_function(
    *,
    file_label: str,
    file_extension: str,
    loading_function: Callable,
    saving_function: Callable,
    labelling_kwargs: Iterable[str],
    cache_folder: Optional[str] = None,
):
    def decorator(fun):
        @wraps(fun)
        def decorated_function(*args, use_cache=True, **kwargs):
            file_name = f"{file_label}"
            for kwarg in labelling_kwargs:
                assert (
                    kwarg in labelling_kwargs
                ), f"{kwarg} is not passed, or not passed as a keyword argument, please check the code."
                file_name += f"_{kwargs}-{kwargs[kwarg]}"

            file_name += f".{file_extension}"

            caching_path = app_static_dir("cache")
            if cache_folder is not None:
                caching_path = caching_path / cache_folder
                caching_path.mkdir(exist_ok=True)

            caching_file = caching_path / file_name

            if caching_file.is_file() and use_cache:
                logger.info(f" loading from cache {caching_file.name}")
                return loading_function(caching_file)
            else:
                returning_value = fun(*args, **kwargs)
                logger.info(f"caching {returning_value} into {caching_file}")
                saving_function(returning_value, caching_file)
                return returning_value

        return decorated_function

    return decorator


def caching_method(
    *,
    file_label,
    file_extension,
    loading_function: Optional[Callable] = None,
    loading_method_ref: Optional[str] = None,
    saving_function: Optional[Callable] = None,
    saving_method_ref: Optional[str] = None,
    labelling_kwargs: Optional[Iterable[str]] = (),
    cache_folder=None,
    object_labelling_attributes=None,
):
    def decorator(fun):

        assert (loading_function is not None) != (loading_method_ref is not None)
        assert (saving_function is not None) != (saving_method_ref is not None)

        @wraps(fun)
        def decorated_method(*args, runtime_labelling="", read_cache=True, write_cache=True, **kwargs):

            file_name = f"{file_label}"

            if runtime_labelling:
                file_name += f"_{runtime_labelling}"

            for kwarg in labelling_kwargs:
                assert (
                    kwarg in labelling_kwargs
                ), f"{kwarg} is not passed, or not passed as a keyword argument, please check the code."
                file_name += f"_{kwargs}-{kwargs[kwarg]}"

            obj = args[0]

            for attribute in object_labelling_attributes:
                file_name += f"_{attribute}-{getattr(obj,attribute)}"

            file_name += f".{file_extension}"

            caching_path = app_static_dir("cache")
            if cache_folder is not None:
                caching_path = caching_path / cache_folder
                caching_path.mkdir(exist_ok=True)

            caching_file = caching_path / file_name

            if loading_function is None:
                loader = getattr(obj, loading_method_ref)
            else:
                loader = loading_function

            if saving_function is None:
                saver = getattr(obj, saving_method_ref)
            else:
                saver = saving_function

            if caching_file.is_file() and read_cache:
                logger.info(f" loading from cache {caching_file.name}")
                return loader(caching_file)
            else:
                returning_value = fun(*args, **kwargs)
                if write_cache:
                    logger.info(f"caching {returning_value} from {fun} into {caching_file}")
                    saver(returning_value, caching_file)
                return returning_value

        return decorated_method

    return decorator
