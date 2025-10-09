import importlib
import pickle
from os import environ, path

import fakeredis
import pytest
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback


def pytest_sessionstart(session):
    """
    Patch MD3 client creation before test modules are imported.
    """
    sc_module = importlib.import_module(
        "mx3_beamline_library.devices.classes.md3.Command.embl.StandardClient"
    )

    sc_module.StandardClient.connect = lambda self: None
    sc_module.StandardClient.isConnected = lambda self: True


@pytest.fixture(scope="session")
def fake_redis():
    return fakeredis.FakeStrictRedis()


@pytest.fixture(scope="session")
def set_bl_active_env():
    environ["BL_ACTIVE"] = "False"
    yield


@pytest.fixture(scope="session")
def sample_id():
    return 1


@pytest.fixture(scope="session")
def run_engine():
    RE = RunEngine({})
    bec = BestEffortCallback()
    RE.subscribe(bec)
    return RE


@pytest.fixture(scope="session")
def session_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp("session_test_directory")


@pytest.fixture(scope="session")
def optical_centering_results():
    try:
        with open(
            path.join(
                path.dirname(__file__), "test_data", "optical_centering_results.pkl"
            ),
            "rb",
        ) as file:
            return pickle.load(file)
    except FileNotFoundError:
        pytest.fail("optical_centering_results.pkl file not found")
    except pickle.UnpicklingError:
        pytest.fail("Error unpickling optical_centering_results.pkl")
