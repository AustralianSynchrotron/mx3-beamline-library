import asyncio
from caproto.server import pvproperty, PVGroup, ioc_arg_parser, run
from mx3_beamline_library.plans.energy_changer.master_energy import MasterEnergy
from mx3_beamline_library.devices.sim.classes.motors import MX3SimMotor

class MasterEnergyIOC(PVGroup):
    master_energy = pvproperty(value=0.0, dtype=float, doc="Master Energy")
    moving = pvproperty(value=0, dtype=int, doc="Moving Status")

    def __init__(self, prefix, *, macros = None, parent = None, name = None):
        super().__init__(prefix, macros=macros, parent=parent, name=name)
        self.bragg_angle_motor = MX3SimMotor(name="bragg_angle_motor")
        self.gap_motor = MX3SimMotor(name="gap_motor")
        self.parallel_translation_motor = MX3SimMotor(name="parallel_translation_motor")

    @master_energy.putter
    async def master_energy(self, instance, value):
        await self.moving.write(1)   
    
        master_energy = MasterEnergy(
            parallel_translation_motor=self.parallel_translation_motor,
            bragg_angle_motor=self.bragg_angle_motor,
            gap_motor=self.gap_motor,
        )
        master_energy.set_master_energy(
            energy=value, harmonic=5, ivu_energy_offset=-0.14, bragg_angle_offset=-0.0542
        )
        # SIMULATE a longer movement. TODO: Remove this line!
        await asyncio.sleep(3)
        await self.moving.write(0)

        return value

if __name__ == "__main__":
    ioc_options, run_options = ioc_arg_parser(
        default_prefix='master_energy:',
        desc='IOC that provides the Master Energy value.'
    )
    ioc = MasterEnergyIOC(**ioc_options)
    run(ioc.pvdb, **run_options)

