import typing
import toml
import mx3_beamline_library
from typing import Generator
from pathlib import Path
from os.path import join, abspath
from mx3_beamline_library import paths, constants
from mx3_beamline_library.devices import _load_sim_devices

if typing.TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


def test_version():
    """Test version detection.

    This shouldn't fail, but the method of package version detection may change.
    Test will fail for versions prior to Python3.8.
    """

    pyproject = abspath(join(Path(__file__).parent, "../pyproject.toml"))
    project_ver = toml.load(pyproject)["tool"]["poetry"]["version"]
    assert mx3_beamline_library.__version__ == project_ver


def test_paths_module():
    """Test "paths" module"""

    assert isinstance(paths.image_path, Path)


def test_constants_module():
    """Test "constants" module"""

    assert isinstance(constants.PV_PREFIX, str)


class TestDevicesInit:
    """Run Devices init tests"""

    base_path = "mx3_beamline_library.devices"
    device_modules = (
        "classes", "detectors", "eps_pss", "io", "motors",
        "optics", "parameters", "sim", "vacuum",
        "flux_capacitors", "neuralyzers",
    )
    device_sim_modules = (
        "classes", "detectors", "eps_pss", "io", "motors",
        "optics", "parameters", "vacuum",
    )

    @classmethod
    def iter_modules_patched(
        cls,
        path: list[str],
        prefix: str = '',
    ) -> Generator[None, str, None]:
        """Patched "iter_modules" method.

        Patches the method to yield module names from a predetermined list.

        Parameters
        ----------
        path : list[str]
            Path to look for modules in.
        prefix : str, optional
            Prefix for module name on output, by default ''

        Yields
        ------
        Generator[None, str, None]
            Generator of device module names.
        """

        res_modules = cls.device_modules
        if Path(path[0]).name == "sim":
            res_modules = cls.device_sim_modules

        for module in res_modules:
            yield None, module, None

    def test_load_sim_devices(self, mocker: "MockerFixture"):
        """ """

        # Patch "iter_modules" and "import_module" methods for future repeatability
        mocker.patch(
            f"{self.base_path}.iter_modules",
            side_effect=self.iter_modules_patched,
        )
        mocker.patch(f"{self.base_path}.import_module", return_value=True)

        # Initialise module list
        modules = {}
        for name in self.device_modules:
            modules[f"{self.base_path}.{name}"] = False

        # Patch system modules
        new_modules: dict = mocker.patch.dict("sys.modules", modules)

        # Load simulated devices
        _load_sim_devices()

        for name in self.device_modules:
            _ignored_modules = ["__init__", "sim", "classes"]
            _module_path = f"{self.base_path}.{name}"
            new_ref = new_modules.get(_module_path)
            old_ref = modules[_module_path]

            if name in _ignored_modules:
                # These module references should not have been modified
                assert new_ref is not None and new_ref == old_ref

            # Check that devices with no sim couterpart module have been removed
            if name in ["flux_capacitors", "neuralyzers"]:
                assert new_ref is None and hasattr(new_modules, name) is False

            # These module references should have been modified
            elif name not in _ignored_modules:
                assert new_ref is True and new_ref != old_ref
