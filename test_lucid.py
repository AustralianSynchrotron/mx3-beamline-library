from os import environ
environ["DECTRIS_DETECTOR_HOST"]= "0.0.0.0"
environ["DECTRIS_DETECTOR_PORT"]= "8000"
environ["BL_ACTIVE"] = "True"
environ["BLUESKY_DEBUG_CALLBACKS"] = "1"
environ["SETTLE_TIME"] = "0.2"

import lucid3
from mx3_beamline_library.plans.basic_scans import grid_scan
from mx3_beamline_library.devices import detectors, motors
from mx3_beamline_library.devices.classes.motors import CosylabMotor
from mx3_beamline_library.devices.classes.detectors import BlackFlyCam

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.plan_stubs import mv, mvr
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

def save_image(data: npt.NDArray,screen_coordinates: list, filename: str):
    plt.imshow(data) 
    plt.scatter(screen_coordinates[1], screen_coordinates[2],s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()

def take_snapshot(camera: BlackFlyCam, filename: str, screen_coordinates: list=[612, 512]):
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )
    plt.imshow(data) 
    plt.scatter(screen_coordinates[0], screen_coordinates[1],s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()

testrig = motors.testrig
motor_x = testrig.x
motor_x.wait_for_connection()
motor_z = testrig.z
motor_z.wait_for_connection()
motor_y = testrig.y
motor_y.wait_for_connection()
motor_phi = testrig.phi
motor_phi.wait_for_connection()

blackfly_camera = detectors.blackfly_camera
blackfly_camera.wait_for_connection()

dectris_detector = detectors.dectris_detector

def plot_raster_grid(camera: BlackFlyCam, initial_pos_pixels: list[int], final_pos_pixels: list[int], filename):
    
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        blackfly_camera.height.get(), blackfly_camera.width.get(), blackfly_camera.depth.get()
    ).astype(np.uint8)

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )
    # Plot loop edge which is used as a reference to plot the grid
    plt.scatter(screen_coordinates[1], screen_coordinates[2],s=200, c="b", marker="+")

    plt.imshow(data)
    
    # Plot Grid limits:
    # Top
    x = np.linspace(initial_pos_pixels[0],final_pos_pixels[0],100)
    z = initial_pos_pixels[1]*np.ones(len(x))
    plt.plot(x,z, color="red", linestyle="--")

    # Bottom
    x = np.linspace(initial_pos_pixels[0],final_pos_pixels[0],100)
    z = final_pos_pixels[1]*np.ones(len(x))
    plt.plot(x,z, color="red", linestyle="--")

    # Right side
    z = np.linspace(initial_pos_pixels[1],final_pos_pixels[1],100)
    x = final_pos_pixels[0]*np.ones(len(x))
    plt.plot(x,z, color="red", linestyle="--")

    # Left side
    z = np.linspace(initial_pos_pixels[1],final_pos_pixels[1],100)
    x = initial_pos_pixels[0]*np.ones(len(x))
    plt.plot(x,z, color="red", linestyle="--")

    plt.savefig(filename)
    plt.close()

def move_motors_to_loop_edge(motor_x: CosylabMotor, motor_z: CosylabMotor, camera: BlackFlyCam):
    beam_position = [612, 512]
    pixels_per_mm = [292.87, 292.87]

    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )
    save_image(data,screen_coordinates,f"figs/fig_{screen_coordinates[1]}")

    loop_position_x = motor_x.position + (screen_coordinates[1] - beam_position[0]) / pixels_per_mm[0]
    loop_position_z = motor_z.position + (screen_coordinates[2] - beam_position[1]) / pixels_per_mm[1]
    print("loop_position_x", loop_position_x)
    print("loop_position_z", loop_position_z)
    yield from mv(motor_x, loop_position_x)
    yield from mv(motor_z, loop_position_z)

def optical_centering(motor_x, motor_z, motor_phi, camera):

    yield from move_motors_to_loop_edge(motor_x, motor_z, camera)
    yield from mvr(motor_phi, 90)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera)
    yield from mvr(motor_phi, 90)
    yield from move_motors_to_loop_edge(motor_x, motor_z, camera)
    take_snapshot(camera, "figs/second_last_pos")

    yield from mvr(motor_z, -0.6)

    take_snapshot(camera, "figs/final_pos")

def master_plan(motor_x: CosylabMotor, motor_z: CosylabMotor, motor_phi: CosylabMotor, camera: BlackFlyCam):
    beam_position = [612, 512]
    pixels_per_mm = [292.87, 292.87]

    yield from optical_centering(motor_x, motor_z, motor_phi, camera)

    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(), camera.width.get(), camera.depth.get()
    )

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )

    delta_z = abs(screen_coordinates[2] - beam_position[1]) / pixels_per_mm[1]

    initial_pos_pixels = [beam_position[0] - abs(screen_coordinates[2] - beam_position[1]),
        beam_position[1] - abs(screen_coordinates[2] - beam_position[1])]
    take_snapshot(
        camera, "figs/initial_pos_grid",
        initial_pos_pixels 
    )
    
    final_pos_pixels = [beam_position[0] + abs(screen_coordinates[2] - beam_position[1]),
        beam_position[1] + abs(screen_coordinates[2] - beam_position[1])]
    take_snapshot(
        camera, "figs/final_pos_grid",
        final_pos_pixels
    )
    plot_raster_grid(camera, initial_pos_pixels, final_pos_pixels,"figs/grid")


    initial_pos_x = motor_x.position - delta_z
    final_pos_x = motor_x.position + delta_z

    initial_pos_z = motor_z.position - delta_z
    final_pos_z = motor_z.position + delta_z

    yield from grid_scan(
        [dectris_detector], motor_z, initial_pos_z, final_pos_z, 2, 
        motor_x, initial_pos_x, final_pos_x, 2,
        md={"sample_id": "test"})



bec = BestEffortCallback()
RE = RunEngine({})
RE.subscribe(bec)

# We will need a low-resolution pin centering to get the loop in the camera view,
# but for now we move the motors to a position where we can see the loop
RE(mv(motor_x, 0))
RE(mv(motor_z, 0))
RE(mv(motor_phi, 0))


print("starting loop centering")

RE(master_plan(motor_x, motor_z, motor_phi, blackfly_camera))





