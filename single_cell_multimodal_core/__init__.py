import logging
import os
import pathlib

import dynaconf

logger = logging.getLogger(__name__)

GLOBAL_ENV_FOR_DYNACONF = "SUGGESTER"
MERGE_ENABLED_FOR_DYNACONF = True
APP_ROOT_DIR = os.getenv(f"{GLOBAL_ENV_FOR_DYNACONF}_ROOT_DIR", str(pathlib.Path(__file__).resolve().parent))
SETTINGS_MODULE_FOR_DYNACONF = "default_settings.yaml,settings.yaml,mirrored_tables_settings.yaml,vertica_table_settings.yaml,suggester_settings.yaml,local_suggester_settings.yaml"
settings = dynaconf.LazySettings(
    GLOBAL_ENV_FOR_DYNACONF=GLOBAL_ENV_FOR_DYNACONF,
    MERGE_ENABLED_FOR_DYNACONF=MERGE_ENABLED_FOR_DYNACONF,
    PROJECT_ROOT_FOR_DYNACONF=APP_ROOT_DIR,
    SETTINGS_MODULE_FOR_DYNACONF=SETTINGS_MODULE_FOR_DYNACONF,
)

credentials = (
    settings.CREDENTIALS if "CREDENTIALS" in settings else logger.warning("No credentials were found to be loaded.")
)
