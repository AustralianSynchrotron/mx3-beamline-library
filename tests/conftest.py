from os import environ

import pytest
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback


@pytest.fixture(scope="session")
def set_bl_active_env():
    environ["BL_ACTIVE"] = "False"
    yield


@pytest.fixture(scope="session")
def sample_id():
    return "test_sample"


@pytest.fixture(scope="session")
def run_engine():
    RE = RunEngine({})
    bec = BestEffortCallback()
    RE.subscribe(bec)
    return RE
