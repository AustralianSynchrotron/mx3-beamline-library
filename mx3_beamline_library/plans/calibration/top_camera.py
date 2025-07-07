import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


from mx3_beamline_library.devices import detectors
from mx3_beamline_library.science.optical_and_loop_centering.loop_edge_detection import LoopEdgeDetection
from mx3_beamline_library.devices.motors import md3
from time import perf_counter
from mx3_beamline_library.schemas.optical_centering import OpticalCenteringExtraConfig
from mx3_beamline_library.devices.motors import md3
from bluesky import RunEngine
from bluesky.plan_stubs import mv
from mx3_beamline_library.plans.plan_stubs import md3_move

config = OpticalCenteringExtraConfig()
roi_x = config.top_camera.roi_x
roi_y = config.top_camera.roi_y

def find_edge(camera):
    img = camera.array_data.get().reshape(camera.height.get(), camera.width.get()).astype(np.uint8)
    img = img[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
    # plt.figure()
    # #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.imshow(img)
    # plt.savefig(f"raw_data")
    # plt.close()
    procImg = LoopEdgeDetection(img, block_size=49, adaptive_constant=6)
    screen_coordinates = procImg.find_tip()

    # save_image(img, screen_coordinates, filename="top_camera")

    return screen_coordinates

def save_image(
    data: npt.NDArray, screen_coordinates: npt.NDArray, filename: str
) -> None:
    """
    Saves an image from a numpy array taken from the camera ophyd object,
    and draws a red cross at the screen_coordinates.

    Parameters
    ----------
    data : npt.NDArray
        A numpy array containing an image from the camera
    x_coord : float
        X coordinate
    y_coord : float
        Y coordinate
    filename : str
        The filename

    Returns
    -------
    None
    """
    x_coord = screen_coordinates[0]
    y_coord = screen_coordinates[1]
    plt.figure()
    plt.imshow(data, cmap='gray', vmin=0, vmax=255)
    plt.scatter(
        x_coord,
        y_coord,
        s=200,
        c="r",
        marker="+",
    )
    #plt.legend()
    plt.savefig(filename)
    plt.close()

def get_x_and_y_coords():
    camera = detectors.blackfly_camera
    camera.wait_for_connection()

    x_vals= []
    y_vals = []
    plt.figure()
    for i in range(30):
        coords = find_edge(camera)
        x_vals.append(coords[0])
        y_vals.append(coords[1])
        plt.scatter(
            coords[0],
            coords[1],
            s=200,
            c="r",
            marker="+",
        )
    data =  camera.array_data.get().reshape(camera.height.get(), camera.width.get()).astype(np.uint8)
    data = data[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]

    y_median = np.median(y_vals)
    #print("x_median", y_median)
    y_std = np.std(y_vals)
    #print("std", y_std)

    x_median = np.median(x_vals)
    #print("y_median", x_median)
    x_std = np.std(x_vals)
    #print("std", x_std)
    return x_median, y_median




def get_pixels_per_y():
    start_alignment_y = 0
    start_sample_x = 0
    start_sample_y = 0
    start_omega = 0
    start_alignment_z = 0
    yield from md3_move(
        md3.omega,start_omega, 
        md3.alignment_y, start_alignment_y, 
        md3.sample_x, start_sample_x, 
        md3.sample_y, start_sample_y,
        md3.alignment_z, start_alignment_z
        )
    start_pixel_x, start_pixel_y= get_x_and_y_coords()
    print("start pos done")

    end_alignment_y = 1
    yield from md3_move(md3.alignment_y, end_alignment_y)
    end_pixel_x, end_pixel_y= get_x_and_y_coords()
    print("end pos done")


    pixels_per_mm_y = abs(start_pixel_y - end_pixel_y) / abs(start_alignment_y - end_alignment_y)
    print("pixels per mm y", pixels_per_mm_y)

def get_pixels_per_x():
    start_alignment_z = 0
    #yield from md3_move(md3.omega, 0, md3.alignment_y, 0, md3.alignment_z, start_alignment_z, md3.sample_y, 0)
    start_alignment_y = 0
    start_sample_x = 0
    start_sample_y = 0
    start_omega = 0
    start_alignment_z = 0
    yield from md3_move(
        md3.omega,start_omega, 
        md3.alignment_y, start_alignment_y, 
        md3.sample_x, start_sample_x, 
        md3.sample_y, start_sample_y,
        md3.alignment_z, start_alignment_z
        )
    start_pixel_x, start_pixel_y= get_x_and_y_coords()
    print("start pos done")

    end_alignment_z  = 1
    yield from md3_move(md3.alignment_z, end_alignment_z)
    end_pixel_x, end_pixel_y= get_x_and_y_coords()
    print("end pos done")


    pixels_per_mm_x = abs(start_pixel_x - end_pixel_x) / abs(start_alignment_z - end_alignment_z)
    print("pixels per mm x", pixels_per_mm_x)

def get_x_and_y_pixels_per_mm():
    yield from get_pixels_per_x()
    yield from get_pixels_per_y()

from bluesky import RunEngine

RE = RunEngine({})

RE(get_x_and_y_pixels_per_mm())