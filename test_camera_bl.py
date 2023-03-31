from os import environ
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from time import sleep

environ["BL_ACTIVE"] = "true"

# NOTE!!!:
# exposure_time = 0.1
# gain = 1
# seems to work on the blackfly camera

from mx3_beamline_library.devices import detectors
from mx3_beamline_library.science.optical_and_loop_centering.psi_optical_centering import loopImageProcessing
from mx3_beamline_library.devices.motors import md3


def save_image(
    data: npt.NDArray, extremes:dict, filename: str
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
    screen_coordinates = extremes[filename]

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



camera = detectors.blackfly_camera
camera.wait_for_connection()

md3.backlight.set(2)
sleep(1)
print("backlight value", md3.backlight.get())
#md3.frontlight.set(0.)
print("frontligh value:", md3.frontlight.get())


def find_edge(camera):
    img = camera.array_data.get().reshape(camera.width.get(), camera.depth.get()).astype(np.uint8)
    procImg = loopImageProcessing(img)
    procImg.findContour(
    zoom="top_camera",
    beamline="MX3",
    )
    extremes = procImg.findExtremes()
    print(extremes["top"])
    save_image(img, extremes, "top")
    return extremes["top"]
"""

x_vals= []
y_vals = []
plt.figure()
for i in range(30):
    coords = find_edge(camera)
    x_vals.append(coords[0])
    y_vals.append(coords[1])
    # plt.imshow(data, cmap='gray', vmin=0, vmax=255)
    plt.scatter(
        coords[0],
        coords[1],
        s=200,
        c="r",
        marker="+",
    )
    #plt.savefig(filename)
data =  camera.array_data.get().reshape(camera.width.get(), camera.depth.get()).astype(np.uint8)

y_median = np.median(y_vals)
print("x_median", y_median)
y_std = np.std(y_vals)
print("std", y_std)

plt.imshow(data, cmap='gray', vmin=0, vmax=255)
plt.savefig("for_loop")
plt.close()


plt.plot(x_vals, label="x coord")
plt.plot(y_vals, label="y coord")

plt.xlabel("iteration")
plt.ylabel("Pixel coordinate")
plt.legend()
plt.savefig("top_values")
"""
# The values below are inferred at omega = 0
start_pixel_y = 395
end_pixel_y = 328
start_motor_y = 0
final_motor_y = 1
pixels_per_mm_y = abs(start_pixel_y - end_pixel_y) / abs(start_motor_y - final_motor_y)
print("pixels_per_mm_y", pixels_per_mm_y)
# result is pixels_per_mm_y =  20.0

# to infer pixels_per_mm_x we set omega=0, and drive sampy
start_pixel_x = 907
start_sampy = 0 
final_sampy = 1
final_pixel_x = 824

pixels_per_mm_x = abs(start_pixel_x - final_pixel_x) / abs(start_sampy - final_sampy)
print("pixels_per_mm_x", pixels_per_mm_x)
# result is pixels_per_mm_x" =  30.0

# This coord for y should be visible in the camera 
y_pixel_target = 472.0
# y_pixel_target = 398
x_pixel_target = 841.0

#md3.alignment_y.position - md3.alignment_y
edge = find_edge(camera)
print(edge)

# Y coords
delta_mm_y = (y_pixel_target - edge[1]) / pixels_per_mm_y
print("delta_mm_y:", delta_mm_y)

# X coords
delta_mm_x = (x_pixel_target - edge[0]) / pixels_per_mm_x
print("delta_mm_x:", delta_mm_x)

if abs(delta_mm_y) < 3:
    md3.alignment_y.set(md3.alignment_y.position - delta_mm_y)
    sleep(1)
    md3.sample_y.set(md3.sample_y.position - delta_mm_x)
else: 
    print("delta pixel > 2")

md3.backlight.set(2)


