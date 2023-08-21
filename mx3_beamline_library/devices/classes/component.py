# from beamline.config import config

import ophyd

def component(fqn: str, device: ophyd.Device, name: str):
    """Creates a component as requested, mediated by beamline configuration.
    """
    #_do_not_subscribe = config().get("do_not_subscribe", [])
    #if fqn in _do_not_subscribe:
    #    return None

    return ophyd.Component(device, name, lazy=True)