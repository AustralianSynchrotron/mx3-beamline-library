# MX3 Beamline Library
This repository hosts the code required for executing data collection plans on the MX3 beamline at the Australian Synchrotron.

## Data Collection Plans
To facilitate data acquisition plans, we rely on [Bluesky](https://github.com/bluesky/bluesky), an experiment orchestration tool. Bluesky interacts with hardware through the [Ophyd](https://github.com/bluesky/ophyd) library, which is a Python-based hardware abstraction layer.

The repository includes various data collection plans, each of which is detailed in the `examples` folder:
- Optical Centering
- X-ray Centering
- Tray Screening

## Ophyd Devices
To execute the plans mentioned above, we provide ophyd-devices for interaction with hardware components. These devices include:
- MD3 Goniometer
- ISARA Robot
- DECTRIS Detector
- Blackfly Camera

## Algorithms
The repository also features algorithms which are used by the optical and x-ray centering plans (see e.g. the `examples` folder):
- Algorithms for crystal identification from raster data
- An algorithm to identify the tip of a loop

## References
We acknowledge the use of external code in this repository:
- The **MD3 exporter client** used for communication is sourced from [MXCUBE](https://github.com/mxcube/mxcubecore).
- The algorithm to identify the tip of the loop is based on code developed by PSI.
