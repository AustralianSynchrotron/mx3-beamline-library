import numpy as np
from os import environ
environ["BL_ACTIVE"] = "True"
environ["DECTRIS_DETECTOR_HOST"] = "0.0.0.0"
environ["DECTRIS_DETECTOR_PORT"] = "8000"

from mx3_beamline_library.schemas.optical_and_xray_centering import RasterGridMotorCoordinates
from mx3_beamline_library.devices.motors import md3  
from mx3_beamline_library.devices.detectors import dectris_detector
from mx3_beamline_library.plans.basic_scans import md3_grid_scan
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plan_stubs import mv
from bluesky.plans import grid_scan
from ophyd.sim import det1
from time import sleep





theta = np.radians(90)

rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
print("rotation matrix: \n", rotation_matrix)

sample_x_sample_y = np.array([0, 1])
print("sample_x and sample_y\n", sample_x_sample_y)
rotated_matrix = np.matmul(sample_x_sample_y, rotation_matrix)
print("rotated matrix \n", rotated_matrix)
#raster_grid_coords = RasterGridMotorCoordinates.parse_obj(result_example)

raster_grid_coords = RasterGridMotorCoordinates(
    initial_pos_sample_x=-0.022731250443299555, 
    final_pos_sample_x=-0.10983893569861315, 
    initial_pos_sample_y=0.6242099418914737, 
    final_pos_sample_y=0.7824280466265174, 
    initial_pos_alignment_y=0.009903480128227239, 
    final_pos_alignment_y=0.43069116007980784, 
    center_pos_sample_x=-0.06628509307095636, 
    center_pos_sample_y=0.7033189942589956, 
    width=0.1806120635408611, 
    height=0.4207876799515806, 
    number_of_columns=1, 
    number_of_rows=2)

def calculate_y_coords(raster_grid_coords: RasterGridMotorCoordinates):
    if raster_grid_coords.number_of_rows == 1:
        # Especial case for number of rows == 1, otherwise
        # we get a division by zero
        motor_positions_array = np.array(
            [np.ones(raster_grid_coords.number_of_columns) * raster_grid_coords.initial_pos_alignment_y]
        )
    
    else:
        delta_y = abs(raster_grid_coords.initial_pos_alignment_y- raster_grid_coords.final_pos_alignment_y) / (raster_grid_coords.number_of_rows-1)
        motor_positions_y = []
        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_y.append(raster_grid_coords.initial_pos_alignment_y + delta_y*i)

        motor_positions_array = np.zeros([raster_grid_coords.number_of_rows,raster_grid_coords.number_of_columns])

        for i in range(raster_grid_coords.number_of_columns):
            if i % 2:
                motor_positions_array[:, i] = np.flip(motor_positions_y)
            else:
                motor_positions_array[:, i] = motor_positions_y

    return motor_positions_array

def calculate_sample_x_coords(raster_grid_coords: RasterGridMotorCoordinates):
    if raster_grid_coords.number_of_columns == 1:
        motor_positions_array = np.array(
            [np.ones(raster_grid_coords.number_of_rows) * raster_grid_coords.initial_pos_sample_x]
        ).transpose()
    else:
        delta = abs(raster_grid_coords.initial_pos_sample_x- raster_grid_coords.final_pos_sample_x) / (raster_grid_coords.number_of_columns-1)
        
        motor_positions = []
        for i in range(raster_grid_coords.number_of_columns):
            motor_positions.append(raster_grid_coords.initial_pos_sample_x - delta*i)

        motor_positions_array = np.zeros([raster_grid_coords.number_of_rows,raster_grid_coords.number_of_columns])

        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_array[i] = motor_positions

    return np.fliplr(motor_positions_array)


def calculate_sample_y_coords(raster_grid_coords: RasterGridMotorCoordinates):
    if raster_grid_coords.number_of_columns == 1:
        motor_positions_array = np.array(
            [np.ones(raster_grid_coords.number_of_rows) * raster_grid_coords.initial_pos_sample_y]
        ).transpose()
    else:
        delta = abs(raster_grid_coords.initial_pos_sample_y- raster_grid_coords.final_pos_sample_y) / (raster_grid_coords.number_of_columns-1)
        
        motor_positions = []
        for i in range(raster_grid_coords.number_of_columns):
            motor_positions.append(raster_grid_coords.initial_pos_sample_y + delta*i)

        motor_positions_array = np.zeros([raster_grid_coords.number_of_rows,raster_grid_coords.number_of_columns])

        for i in range(raster_grid_coords.number_of_rows):
            motor_positions_array[i] = motor_positions

    return np.fliplr(motor_positions_array)

y_array = calculate_y_coords(raster_grid_coords)
sample_x_array = calculate_sample_x_coords(raster_grid_coords)
sample_y_array = calculate_sample_y_coords(raster_grid_coords)


print("y axis array:\n", y_array)
print("sample_x array:\n", sample_x_array)
print("sample_y array:\n", sample_y_array)

RE = RunEngine({})
bec = BestEffortCallback()
RE.subscribe(bec)

def my_grid_scan_plan(raster_grid_coords: RasterGridMotorCoordinates):
    #yield from mv
    y_array = calculate_y_coords(raster_grid_coords)
    sample_x_array = calculate_sample_x_coords(raster_grid_coords)
    sample_y_array = calculate_sample_y_coords(raster_grid_coords)
    for j in range(raster_grid_coords.number_of_columns):
        for i in range(raster_grid_coords.number_of_rows):
            print(
                "alignment_x:", round(y_array[i,j],1), ",sample_x: ", round(sample_x_array[i,j],1), 
                "sample_y:", round(sample_y_array[i,j],1))
            yield from mv(
                md3.alignment_y, y_array[i,j],
                md3.sample_x, sample_x_array[i,j],
                md3.sample_y, sample_y_array[i,j]
                )
            sleep(0.5)
            
scan_type = "my_plan"

if scan_type == "my_plan":
    RE(my_grid_scan_plan(raster_grid_coords))
elif scan_type == "bluesky":
    RE(
    grid_scan(
        [det1], 
        md3.sample_x, raster_grid_coords.initial_pos_sample_x, raster_grid_coords.final_pos_sample_x, raster_grid_coords.number_of_columns,
        md3.sample_y, raster_grid_coords.initial_pos_sample_y, raster_grid_coords.final_pos_sample_y, raster_grid_coords.number_of_columns,
        md3.alignment_y, raster_grid_coords.initial_pos_alignment_y, raster_grid_coords.final_pos_alignment_y, raster_grid_coords.number_of_rows
        )
   )
elif scan_type == "md3":

    RE(
    md3_grid_scan(
                    detector=dectris_detector,
                    detector_configuration={"nimages:": 1}, # this is not used
                    metadata={"sample_id": "sample_test"},
                    grid_width= raster_grid_coords.width,
                    grid_height= raster_grid_coords.height,
                    number_of_columns= raster_grid_coords.number_of_columns,
                    number_of_rows= raster_grid_coords.number_of_rows,
                    start_omega= 331.16,
                    start_alignment_y=raster_grid_coords.initial_pos_alignment_y,
                    start_alignment_z=0.629,
                    start_sample_x=raster_grid_coords.final_pos_sample_x,
                    start_sample_y=raster_grid_coords.final_pos_sample_y,
                    exposure_time=5,
                    use_fast_mesh_scans=True
)
)

right_corner = {"sample_x": -0.109, "sample_y": 0.782, "alignment_y": 0.43}
