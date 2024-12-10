from os import environ

import pytest


@pytest.fixture(scope="session")
def set_bl_active_env():
    environ["BL_ACTIVE"] = "False"
    yield
