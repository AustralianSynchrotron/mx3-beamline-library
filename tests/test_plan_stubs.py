import pytest

from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.plans.plan_stubs import (
    md3_move,
    set_actual_sample_detector_distance,
    set_distance_and_md3_phase,
    set_distance_and_transmission,
)


def test_md3_move(run_engine):
    # Exercise
    run_engine(
        md3_move(
            md3.omega,
            0,
            md3.sample_x,
            0,
            md3.sample_y,
            0,
            md3.alignment_x,
            0,
            md3.alignment_y,
            0,
        )
    )

    # Verify
    assert md3.omega.position == 0
    assert md3.sample_x.position == 0
    assert md3.sample_y.position == 0
    assert md3.alignment_x.position == 0
    assert md3.alignment_y.position == 0


def test_set_actual_sample_detector_distance(run_engine):
    # Exercise
    result = run_engine(set_actual_sample_detector_distance(50))

    # Verify
    assert result == ()


def test_set_actual_sample_detector_distance_limit_failure(run_engine):
    # Exercise and verify
    with pytest.raises(ValueError):
        run_engine(set_actual_sample_detector_distance(500000))


def test_set_distance_and_phase(run_engine):
    # Exercise
    result = run_engine(set_distance_and_md3_phase(100, "Transfer"))

    # Verify
    assert result == ()


def test_set_distance_and_transmission(run_engine):
    # Exercise
    result = run_engine(set_distance_and_transmission(100, 0.5))

    # Verify
    assert result == ()


@pytest.mark.parametrize("transmission", [1.1, -1])
def test_set_transmission_error(run_engine, transmission):
    with pytest.raises(ValueError):
        run_engine(set_distance_and_transmission(500, transmission))
