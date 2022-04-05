from as_beamline_library.devices import motors


def test_motor():
    """Test motor."""

    test_pos_1 = 10
    motors.my_table.x.set(test_pos_1)
    assert motors.my_table.x.position == test_pos_1


if __name__ == "__main__":
    test_motor()
