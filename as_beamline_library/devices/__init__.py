""" Files in the top level of ``devices`` contain the instantiation of devices. The names of
    these files is at the discretion of Scientific Computing and Beamline Staff. However some
    suggested files appear in the template. Do not prepend these file names with beamline
    name/abbreviation.

    Imports are conditional on an environment variable ``BL_ACTIVE``. Sim devices are imported if
    not ``True``.
"""

import importlib
from os import environ
from pathlib import Path
from pkgutil import iter_modules
from sys import modules

IMPORT_PATH = Path(__file__).parent


def _real_sim_switch():
    """."""
    try:
        if environ["BL_ACTIVE"] == "True":
            return
    except KeyError:
        pass

    import_dict = {}

    for _, name, _ in iter_modules([(IMPORT_PATH / "sim").as_posix()]):
        if name in ["__init__", "sim", "classes"]:
            continue

        modules[f"as_beamline_library.devices.{name}"] = import_dict[
            f"{name}"
        ] = importlib.import_module(
            f".{name}", package="as_beamline_library.devices.sim"
        )


_real_sim_switch()
