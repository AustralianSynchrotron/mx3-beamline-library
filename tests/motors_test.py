from as_beamline_library.devices.motors import my_table

def test_motor():
    
    test_pos_1 = 10
    my_table.x.set(test_pos_1)
    assert my_table.x.position == test_pos_1

