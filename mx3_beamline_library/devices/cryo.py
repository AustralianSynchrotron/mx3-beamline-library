from ophyd import EpicsSignalRO

cryo_temperature = EpicsSignalRO("MX3CRYOJET01:SAMPLET_MON", name="cryo_temperature")
