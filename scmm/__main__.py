"""
todo: temporary main to be refactored
-> description: add argparse for command-line (CL) configuration set-up

-> usage:
    a) python main_tmp.py configuration_file
    b) python main_tmp.py config_name

-> logic:
    1) retrieve a configuration name from the CL
    2) check if the configuration name is present on the config dictionary
       else check if the configuration name is a file on the config folder
    4) run the pipline with the proper config or rise an error
"""

import click
import importlib
import logging

import pandas as pd

from scmm.utils.appdirs import app_static_dir
from scmm.problems.cite.configurations import config_dict as cite_configs
from scmm.problems.multiome.configurations import config_dict as multiome_configs
from scmm.utils.log import setup_logging

logger = logging.getLogger(__name__)


def load_submission(problem, name):
    return pd.read_csv(app_static_dir(f"out/{problem}") / f"{name}.csv", index_col="row_id").squeeze(
        "columns"
    )


def load_model(problem, name, configs):
    config_file = configs.get(name, name)
    config_module = importlib.import_module(f"scmm.problems.{problem}.configurations." + config_file)
    configuration = config_module.configuration
    model_class = config_module.model_class

    model_wrapper = model_class(configuration=configuration, label=config_module.model_label)

    return model_wrapper


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    # logging setup
    setup_logging("DEBUG" if debug else "INFO")

    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@cli.command("submission")
@click.argument('cite')
@click.argument('multiome')
def submission(cite: str, multiome: str):
    def get_submission(problem, name, configs):
        if (app_static_dir(f"out/{problem}") / f"{name}.csv").exists():
            output = load_submission(problem, name)
        else:
            model = load_model(problem, name, configs)
            logger.info(f"starting full pipeline for problem with configuration: {model.model_label}")
            output = model.full_pipeline(refit=True)
        return output

    subs = []
    for problem, name, config in zip(['cite', 'multiome'], [cite, multiome], [cite_configs, multiome_configs]):
        sub = get_submission(problem, name, config)
        sub.to_csv(app_static_dir(f"out/{problem}") / f"{name}.csv")
        subs.append(sub)

    full = pd.concat(subs, axis=0)
    full.to_csv(app_static_dir("out/full_submission") / f"{cite}-{multiome}.csv")


if __name__ == "__main__":
    cli()