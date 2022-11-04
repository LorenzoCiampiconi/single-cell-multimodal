import importlib
import logging

import click
import pandas as pd

from scmm.problems.cite.configurations import config_dict as cite_configs
from scmm.problems.multiome.configurations import config_dict as multiome_configs
from scmm.utils.appdirs import app_static_dir

logger = logging.getLogger(__name__)


all_configs = {
    "cite": cite_configs,
    "multiome": multiome_configs,
}


def load_submission(problem, name):
    return pd.read_csv(app_static_dir(f"out/{problem}") / f"{name}.csv", index_col="row_id").squeeze("columns")


def build_model(problem, name, configs):
    if name not in configs:
        logger.warn(f"Add the configutration to the proper dict please! Dynamic loading will be deprecated shortly")
        config_module = importlib.import_module(f"scmm.problems.{problem}.configurations." + name)
    else:
        config_module = configs[name]

    configuration = config_module.configuration
    model_class = config_module.model_class

    model_wrapper = model_class(configuration=configuration, label=config_module.model_label)

    return model_wrapper


def get_submission(problem, name, configs):
    if (app_static_dir(f"out/{problem}") / f"{name}.csv").exists():
        output = load_submission(problem, name)
    else:
        model = build_model(problem, name, configs)
        if model.saved_model.exists():
            model = model.load_model()
        else:
            click.confirm(
                'No model available, training needed... Should I just "fit"? (check help for other commands)',
                abort=True,
            )
            model.fit_model()
        output = model.predict_public()
        output.to_csv(app_static_dir(f"out/{problem}") / f"{name}.csv")

    return output


def save_submissions(cite, multiome):
    subs = []
    for problem, name in zip(["cite", "multiome"], [cite, multiome]):
        sub = get_submission(problem, name, all_configs[problem])
        subs.append(sub)

    full = pd.concat(subs, axis=0)
    full.to_csv(app_static_dir("out/full_submissions") / f"{cite}-{multiome}.csv")


def run_crossvalidation(problem, model_name):
    model = build_model(problem, model_name, all_configs[problem])
    model.full_pipeline(refit=False)
