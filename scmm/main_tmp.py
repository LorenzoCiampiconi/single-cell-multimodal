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

import importlib
import argparse
import logging
from scmm.utils.log import setup_logging
from scmm.problems.cite.concrete import LGBMwMultilevelEmbedderCite
from scmm.problems.cite.configurations.config_dict import config_dic

if __name__ == "__main__":
    # logging setup
    setup_logging("DEBUG")
    logger = logging.getLogger(__name__)

    # argparse setup
    parser = argparse.ArgumentParser(description="scmm full pipline")
    parser.add_argument("config_name", type=str, help="configuration name or configuration file name")
    args = parser.parse_args()

    if args.config_name in config_dic:
        config_file = config_dic.get(args.config_name)
    else:
        config_file = args.config_name

    config_module = importlib.import_module("scmm.problems.cite.configurations." + config_file)
    logger.info("starting full pipeline with configuration: " f"{config_module.model_label}")

    model_wrapper = LGBMwMultilevelEmbedderCite(
        configuration=config_module.configuration, label=config_module.model_label
    )
    model_wrapper.full_pipeline()
