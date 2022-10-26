import logging
from functools import wraps
from pathlib import Path
from typing import Iterable

import numpy as np

from scmm.utils.appdirs import app_static_dir

logger = logging.getLogger(__name__)


def np_load_wrapper(svd_caching_path):
    with np.load(svd_caching_path) as npz_file:
        train_mat, test_mat = npz_file["train_mat"], npz_file["test_mat"]
        return train_mat, test_mat


def caching_function(
    *, file_label, file_extension, loading_function, saving_function, labelling_kwargs: Iterable[str], cache_folder=None, object_labelling_attributes=None
):
    def decorator(fun):
        @wraps(fun)
        def decorated_function(*args, use_cache=True, **kwargs):
            file_name = f"{file_label}"
            for kwarg in labelling_kwargs:
                # assert (
                #     kwarg in labelling_kwargs
                # ), f"{kwarg} is not passed, or not passed as a keyword argument, please check the code."
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

            if caching_file.is_file() and use_cache:
                logger.info(f" loading from cache {caching_file.name}")
                return loading_function(caching_file)
            else:
                returning_value = fun(*args, **kwargs)
                logger.info(f'caching {returning_value} into {caching_file}')
                saving_function(returning_value, caching_file)
                return returning_value

        return decorated_function

    return decorator
