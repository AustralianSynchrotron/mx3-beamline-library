import uuid
from os import environ, path

import numpy as np
import pytest
from bluesky import RunEngine
from ophyd.sim import det1, motor1, motor2

motor1.delay = 0


def test_1d_scan_gaussian_distribution(run_engine: RunEngine, session_tmpdir):
    # Setup
    # Set HDF5_OUTPUT_DIRECTORY env variable before
    # importing the beamline library
    environ["HDF5_OUTPUT_DIRECTORY"] = str(session_tmpdir)
    from mx3_beamline_library.devices.sim.classes.signals import SimCameraStats
    from mx3_beamline_library.plans.commissioning.commissioning import Scan1D
    from mx3_beamline_library.plans.commissioning.stats import ScanStats1D

    total = SimCameraStats(name="total", distribution="gaussian")
    mean = SimCameraStats(name="mean", distribution="gaussian")

    hdf5_filename = f"gaussian_{uuid.uuid4().hex}.h5"

    # Execute
    scan_1d = Scan1D(
        [total, mean],
        motor1,
        -1,
        1,
        7,
        hdf5_filename=hdf5_filename,
        calculate_first_derivative=False,
    )
    run_engine(scan_1d.run())

    # Verify
    assert path.exists(path.join(session_tmpdir, hdf5_filename))
    assert isinstance(scan_1d.statistics, list)
    assert len(scan_1d.statistics) == 2
    assert isinstance(scan_1d.statistics[0], ScanStats1D)
    assert isinstance(scan_1d.statistics[1], ScanStats1D)
    assert isinstance(scan_1d.intensity_dict["total"], list)
    assert isinstance(scan_1d.intensity_dict["mean"], list)
    assert isinstance(scan_1d.motor_array, np.ndarray)
    assert list(scan_1d.metadata.keys()) == [
        "hdf5_filename",
        "favourite",
        "favourite_description",
    ]
    assert isinstance(scan_1d.metadata["hdf5_filename"], str)
    assert isinstance(scan_1d.metadata["favourite"], bool)
    assert isinstance(scan_1d.metadata["favourite_description"], str)


def test_1d_scan_smooth_step_distribution(run_engine: RunEngine, session_tmpdir):
    # Setup
    environ["HDF5_OUTPUT_DIRECTORY"] = str(session_tmpdir)
    from mx3_beamline_library.devices.sim.classes.signals import SimCameraStats
    from mx3_beamline_library.plans.commissioning.commissioning import Scan1D
    from mx3_beamline_library.plans.commissioning.stats import ScanStats1D

    total = SimCameraStats(name="total", distribution="smooth_step")
    mean = SimCameraStats(name="mean", distribution="smooth_step")

    hdf5_filename = f"smooth_step_{uuid.uuid4().hex}.h5"

    # Execute
    scan_1d = Scan1D(
        [total, mean],
        motor1,
        -1,
        1,
        7,
        hdf5_filename=hdf5_filename,
        calculate_first_derivative=True,
    )
    run_engine(scan_1d.run())

    # Verify
    assert isinstance(scan_1d.first_derivative, np.ndarray)
    assert isinstance(scan_1d.statistics, list)
    assert len(scan_1d.statistics) == 2
    assert isinstance(scan_1d.statistics[0], ScanStats1D)
    assert isinstance(scan_1d.statistics[1], ScanStats1D)


def test_scan_1d_inverted_gaussian(run_engine, session_tmpdir):
    # Setup
    environ["HDF5_OUTPUT_DIRECTORY"] = str(session_tmpdir)
    from mx3_beamline_library.devices.sim.classes.signals import SimCameraStats
    from mx3_beamline_library.plans.commissioning.commissioning import Scan1D

    total = SimCameraStats(name="total", distribution="smooth_step", flip=True)
    mean = SimCameraStats(name="mean", distribution="smooth_step", flip=True)

    hdf5_filename = f"smooth_step_{uuid.uuid4().hex}.h5"

    # Execute
    scan_1d = Scan1D(
        [total, mean],
        motor1,
        -1,
        1,
        7,
        hdf5_filename=hdf5_filename,
        calculate_first_derivative=True,
    )
    run_engine(scan_1d.run())

    # Verify
    assert scan_1d._flipped_gaussian is True


def test_scan_2d(run_engine, session_tmpdir):
    environ["HDF5_OUTPUT_DIRECTORY"] = str(session_tmpdir)
    from mx3_beamline_library.plans.commissioning.commissioning import Scan2D

    scan_2d = Scan2D(
        detectors=[det1],
        motor_1=motor1,
        initial_position_motor_1=-1,
        final_position_motor_1=100,
        number_of_steps_motor_1=5,
        motor_2=motor2,
        initial_position_motor_2=-1,
        final_position_motor_2=100,
        number_of_steps_motor_2=5,
    )
    run_engine(scan_2d.run())
    assert isinstance(scan_2d.intensity, dict)
    assert isinstance(scan_2d.intensity["det1"], list)
    assert list(scan_2d.metadata.keys()) == [
        "hdf5_filename",
        "favourite",
        "favourite_description",
    ]
    assert isinstance(scan_2d.metadata["hdf5_filename"], str)
    assert isinstance(scan_2d.metadata["favourite"], bool)
    assert isinstance(scan_2d.metadata["favourite_description"], str)


@pytest.mark.parametrize("motor_limits", [(-11, 5), (-5, 20)])
def test_check_motor_limits_failure(motor_limits):
    from mx3_beamline_library.plans.commissioning.commissioning import (
        _check_motor_limits,
    )

    motor1.limits = [-10.0, 10.0]
    with pytest.raises(ValueError):
        _check_motor_limits(motor1, motor_limits[0], motor_limits[1])


@pytest.mark.parametrize("motor_limits", [(-5, 5), (-10, 10)])
def test_check_motor_limits(motor_limits):
    from mx3_beamline_library.plans.commissioning.commissioning import (
        _check_motor_limits,
    )

    motor1.limits = [-10.0, 10.0]
    result = _check_motor_limits(motor1, motor_limits[0], motor_limits[1])

    assert result is None
