""" Files in the top level of ``devices`` contain the instantiation of devices. The names of
    these files is at the discretion of Scientific Computing and Beamline Staff. However, some
    suggested files appear in the template. Do not prepend these file names with the beamline
    name/abbreviation.

    Imports are conditional on an environment variable ``BL_ACTIVE``. The real device modules will
    *not* be imported unless this is set to True. If present the sim device modules will be imported
    if not ``True``.
"""

import importlib
from os import environ
from pathlib import Path
from pkgutil import iter_modules
from sys import modules

IMPORT_PATH = Path(__file__).parent


def _real_sim_switch():
    """This function loads the real device modules if the environment variable ``BL_ACTIVE`` is
    True, otherwise sim device modules are loaded."""

    try:
        if environ["BL_ACTIVE"].lower() == "true":
            return
    except KeyError:
        pass

    import_dict = {}

    sim_modules = set()
    for _, name, _ in iter_modules([(IMPORT_PATH / "sim").as_posix()]):
        if name in ["__init__", "classes"]:
            continue

        sim_modules.add(name)

        modules[f"as_beamline_library.devices.{name}"] = import_dict[
            f"{name}"
        ] = importlib.import_module(
            f".{name}", package="as_beamline_library.devices.sim"
        )

    for _, name, _ in iter_modules([(IMPORT_PATH).as_posix()]):
        if name in ["__init__", "sim", "classes"] + list(sim_modules):
            continue
        del modules[f"as_beamline_library.devices.{name}"]


_real_sim_switch()
