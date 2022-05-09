from bluesky.utils import Msg, root_ancestor, separate_devices


def stage(devices):
    devices = separate_devices(root_ancestor(device) for device in devices)
    for d in devices:
        yield Msg("stage", d)


def unstage(devices):
    devices = separate_devices(root_ancestor(device) for device in devices)

    for d in reversed(devices):
        yield Msg("unstage", d)
