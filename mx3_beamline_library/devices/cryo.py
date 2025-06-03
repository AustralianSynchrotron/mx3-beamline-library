from ophyd import EpicsSignal

cryo_temperature = EpicsSignal(read_pv="MX3CRYOJET01:SAMPLET_MON", write_pv="MX3CRYOJET01:SETP_CMD", name="cryo_temperature")

