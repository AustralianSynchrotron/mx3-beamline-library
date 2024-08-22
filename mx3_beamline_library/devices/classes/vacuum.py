from ophyd import Component as Cpt, Device, EpicsSignalRO, Kind

from . import Register


@Register("Cold Cathode Gauge")
class ColdCathodeGaugeController(Device):
    pressure_monitor = Cpt(
        EpicsSignalRO,
        ":PRESSURE_MONITOR",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="IP calc pressure",
    )
    status = Cpt(
        EpicsSignalRO,
        ":STATUS",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="IP run status",
    )


@Register("Ion Pump")
class IonPumpController(Device):
    pressure_monitor = Cpt(
        EpicsSignalRO,
        ":PRESSURE_MONITOR",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="IP calc pressure",
    )
    status = Cpt(
        EpicsSignalRO,
        ":STATUS",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="IP run status",
    )
    current_monitor = Cpt(
        EpicsSignalRO,
        ":CURRENT_MONITOR",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="IP current readback (uA)",
    )
    current_monitor_egu = Cpt(
        EpicsSignalRO,
        ":CURRENT_MONITOR.EGU",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="IP current units readback",
    )
    voltage_monitor = Cpt(
        EpicsSignalRO,
        ":VOLTAGE_MONITOR",
        kind=Kind.omitted,
        auto_monitor=True,
        doc="IP HV readback (kV)",
    )
