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

import ast
from importlib import import_module
from os import environ
from pathlib import Path
from pkgutil import iter_modules
from sys import modules
import logging

from ophyd.sim import instantiate_fake_device, make_fake_device

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

        modules[f"mx3_beamline_library.devices.{name}"] = import_dict[
            f"{name}"
        ] = import_module(f".{name}", package="mx3_beamline_library.devices.sim")

    for _, name, _ in iter_modules([(IMPORT_PATH).as_posix()]):
        if name in ["__init__", "sim", "classes"] + list(sim_modules):
            continue
        del modules[f"mx3_beamline_library.devices.{name}"]


class InstDef:
    """."""

    def __init__(self, i_class, args, kwargs):
        self._i_class = i_class
        self._args = args
        self._kwargs = kwargs

    @property
    def i_class(self):
        return self._i_class

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs


class FormattedValueError(Exception):
    """Formatted Value Error"""


def get_string_value(arg, constants: dict[str] | None):
    """."""
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Str):
        return arg.value
    if isinstance(arg, ast.FormattedValue):
        if arg.value.id in constants:
            return constants[arg.value.id]
        else:
            raise FormattedValueError(arg.value.id)

    if isinstance(arg, ast.Constant):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        val = ""
        for v in arg.values:
            val += get_string_value(v, constants=constants)
        return val


class VistDevices(ast.NodeVisitor):
    """."""

    def __init__(self):
        self.imports = {}
        self.instances = {}

    def visit_Import(self, node):
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module.split(".")[0] not in ["sims"]:
            for alias in node.names:
                parent = ""
                if node.level == 1:
                    parent = ".".join(__name__.split(".")) + "."
                if node.level > 1:
                    parent = ".".join(__name__.split(".")[: 1 - node.level]) + "."
                self.imports[alias.name] = f"{parent}{node.module}"
        self.generic_visit(node)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):

            vals = []

            done = False
            constants = {}
            while not done:
                try:
                    for arg in node.value.args:
                        vals.append(get_string_value(arg, constants=constants))
                except FormattedValueError as e:
                    module = import_module(self.imports[e.args[0]])
                    constants[e.args[0]] = getattr(module, e.args[0])
                    continue

                done = True

            kw_dict = {}

            done = False
            while not done:
                try:
                    for kwarg in node.value.keywords:
                        kw_dict[kwarg.arg] = get_string_value(
                            kwarg.value, constants=constants
                        )
                except FormattedValueError as e:
                    module = import_module(self.imports[e.args[0]])
                    constants[e.args[0]] = getattr(module, e.args[0])
                    continue

                done = True

            self.instances[node.targets[0].id] = InstDef(
                node.value.func.id, vals, kw_dict
            )

        self.generic_visit(node)


def _auto_faker():

    for ff, module, _ in iter_modules([(IMPORT_PATH).as_posix()]):
        if module in ["__init__", "sim", "classes"]:
            continue
        with open(f"{ff.path}/{module}.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        v = VistDevices()
        v.visit(tree)
        for name, defn in v.instances.items():
            class_module = import_module(v.imports[defn.i_class])
            try:
                device = instantiate_fake_device(
                    getattr(class_module, defn.i_class),
                    prefix=defn.args[0],
                    **defn.kwargs,
                )
            except TypeError:
                try:
                    device = make_fake_device(getattr(class_module, defn.i_class))(
                        defn.args
                    )
                except TypeError:
                    pass
            device_module = import_module(f"{__name__}.{module}")
            if not hasattr(device_module, name):
                setattr(device_module, name, device)


try:
    if environ["BL_ACTIVE"].lower() == "false":
        _load_sim_devices()
        logging.info("Using simulated devices")
    else:
        logging.info("Using real devices")
except KeyError:
    _load_sim_devices()
    logging.info("Using simulated devices")

try:
    if environ["AUTO_FAKE"].lower() == "true":
        _auto_faker()
        logging.info("AUTO_FAKE=True")
    else:
        logging.info("AUTO_FAKE=False")
except KeyError:
    logging.info("AUTO_FAKE=False")
