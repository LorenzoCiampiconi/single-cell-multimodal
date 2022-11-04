import logging

import click

from scmm.logic import run_crossvalidation, save_submissions
from scmm.utils.log import setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug", "log_level", flag_value="DEBUG")
@click.option("--info", "log_level", flag_value="INFO", default=True)
def cli(log_level):
    setup_logging(log_level)

    click.echo(f"Log level: {log_level}")


@cli.command()
@click.argument("cite")
@click.argument("multiome")
def submission(cite: str, multiome: str):
    save_submissions(cite, multiome)


@cli.command()
@click.argument("problem", type=click.Choice(["cite", "multiome"], case_sensitive=False))
@click.argument("model_name")
def crossval(problem: str, model_name: str):
    run_crossvalidation(problem, model_name)
