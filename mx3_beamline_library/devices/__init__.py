""" Files in the top level of ``devices`` contain the instantiation of devices. The names of
    these files is at the discretion of Scientific Computing and Beamline Staff. However, some
    suggested files appear in the template. Do not prepend these file names with the beamline
    name/abbreviation.

    Imports are conditional on an environment variable ``BL_ACTIVE``.
    The real device modules will *not* be imported unless this is set to True.
    If present the sim device modules will be imported
    if not ``True``.
"""
from __future__ import annotations

from importlib import import_module
from os import environ
from pathlib import Path
from pkgutil import iter_modules
from sys import modules
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

IMPORT_PATH = Path(__file__).parent


def _load_sim_devices():
    """This function loads the sim device modules instead of the real device modules."""

    import_dict = {}

    sim_modules = set()
    for _, name, _ in iter_modules([(IMPORT_PATH / "sim").as_posix()]):
        if name in ["__init__", "classes"]:
            continue

        sim_modules.add(name)

        modules[f"{__package__ or __name__}.{name}"] = import_dict[
            f"{name}"
        ] = import_module(f".{name}", package=f"{__package__ or __name__}.sim")

    for _, name, _ in iter_modules([(IMPORT_PATH).as_posix()]):
        if name in ["__init__", "sim", "classes"] + list(sim_modules):
            continue
        del modules[f"{__package__ or __name__}.{name}"]


try:
    if environ["BL_ACTIVE"].lower() == "false":
        _load_sim_devices()
        logging.info("Using simulated devices")
    else:
        logging.info("Using real devices")
except KeyError:
    _load_sim_devices()
    logging.info("Using simulated devices")
