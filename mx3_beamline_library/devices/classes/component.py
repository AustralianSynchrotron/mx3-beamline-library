import datetime
import importlib
import json
import os
from functools import cache
from typing import TypeAlias, Union

import httpx
import httpx_file
import ophyd

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

BITBUCKET_CONFIG_URL = os.environ.get(
    "BITBUCKET_CONFIG_URL",
    r"https://bitbucket.synchrotron.org.au/projects/MX3/repos/mx3-config/raw/config.json?at=refs%2Fheads%2Fmain",  # noqa
)


@cache
def _config() -> dict[str, "JSON"]:
    """Retrieves the current beamline configuration from source
    control and returns it as a JSON dictionary. If the network
    connection fails, returns the configuration as was available at
    build time.

    The keys and values are up to the beamline scientists. Keys
    beginning with an underscore denote metadata.

    The function is cached, to avoid doing lots of network calls, but
    also to provide a consistent view of the configuration during the
    process. To reset the cache, restart the process.

    For debug/testing/mocking purposes, will look for the environment
    variable BSX_BEAMLINE_CONFIG_URL, which can be a file:// URL.

    """
    client = httpx.Client(mounts={"file:/": httpx_file.FileTransport()})

    try:
        r = client.get(BITBUCKET_CONFIG_URL)
        j = r.json()
    except (httpx.ConnectError, httpx.ReadError, json.decoder.JSONDecodeError) as e:
        p = importlib.resources.as_file(
            importlib.resources.files("beamline") / "resources" / "config.json"
        )
        with p as f:
            j = json.load(f.open())
            j["_meta"]["reason"] = str(e)

    else:
        j["_meta"] = {
            "build_time_resource": False,
            "fetched": datetime.datetime.now().isoformat(),
        }

    return j


def component(
    fqn: str, device: ophyd.Device, name: str
) -> Union[ophyd.Component, None]:
    """
    Creates a component as requested, mediated by beamline configuration.

    Parameters
    ----------
    fqn : str
        The fully qualified name of the device, e.g. mx3.pds.pfm.deltaq_cs
    device : ophyd.Device
        The device class, e.g. EpicsMotor
    name : str
        The PV prefix

    Returns
    -------
    Union[ophyd.Component, None]
        Returns None if the name of a the device is in the do_not_subscribe entry
        of the configuration file, otherwise returns an Ophyd Component.
    """
    _do_not_subscribe = _config().get("do_not_subscribe", [])
    if fqn in _do_not_subscribe:
        return None

    return ophyd.Component(device, name, lazy=True)
