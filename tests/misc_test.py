import toml
import mx3_beamline_library
from pathlib import Path
from os.path import join, abspath
from mx3_beamline_library import paths, constants

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
