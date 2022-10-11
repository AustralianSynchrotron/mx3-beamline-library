import cv2
from os import environ  # noqa
import numpy.typing as npt
import numpy as np
import pickle
import matplotlib.pyplot as plt
import lucid3

environ["DECTRIS_DETECTOR_HOST"] = "0.0.0.0"  # noqa
environ["DECTRIS_DETECTOR_PORT"] = "8000"  # noqa
environ["BL_ACTIVE"] = "True"  # noqa
environ["BLUESKY_DEBUG_CALLBACKS"] = "1"  # noqa
environ["SETTLE_TIME"] = "0.2"  # noqa

import logging


from mx3_beamline_library.devices import detectors, motors
from mx3_beamline_library.devices.classes.detectors import BlackFlyCam


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

camera = detectors.blackfly_camera
camera.wait_for_connection()

testrig = motors.testrig
motor_x = testrig.x
motor_x.wait_for_connection()
motor_z = testrig.z
motor_z.wait_for_connection()
motor_y = testrig.y
motor_y.wait_for_connection()
motor_phi = testrig.phi
motor_phi.wait_for_connection()

motor_phi.move(180, wait=True)
motor_x.move(1, wait=True)
motor_z.move(0, wait=True)
motor_y.move(0.0, wait=True)

def save_image(
    data: npt.NDArray, screen_coordinates: list[int, int, int], filename: str
) -> None:
    """
    Saves an image from a numpy array taken from the camera ophyd object,
    and draws a red cross at the screen_coordinates.

    Parameters
    ----------
    data : npt.NDArray
        A numpy array containing an image from the camera
    screen_coordinates : list
        A list containing lucid3 results
    filename : str
        The filename

    Returns
    -------
    None
    """
    plt.figure()
    plt.imshow(data)
    plt.scatter(screen_coordinates[1], screen_coordinates[2], s=200, c="r", marker="+")
    plt.savefig(filename)
    plt.close()


def calculate_laplacian_variance(camera: BlackFlyCam):
    array_data: npt.NDArray = camera.array_data.get()
    data = array_data.reshape(
        camera.height.get(),
        camera.width.get(),
        camera.depth.get(),
    ).astype(np.uint8)

    screen_coordinates = lucid3.find_loop(
        image=data,
        rotation=True,
        rotation_k=1,
    )
    print(screen_coordinates)

    if screen_coordinates[0] == "No loop detected":
        return None

    save_image(
        data,
        screen_coordinates,
        f"figs/step_2_loop_centering_fig_{screen_coordinates[1]}",
        )


    gray_image = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('gray_image.jpg', gray_image)

    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

# motor_y_vals = np.linspace(-1, 1, 20)

laplacian_variance = []
motor_position = []
laplacian_variance.append(calculate_laplacian_variance(camera))
motor_position.append(motor_y.position)

step_size = 0.1
tolerance = 4
diff = None

if diff is None:
    motor_y.move(motor_y.position + step_size, wait=True)
    laplacian_variance.append(calculate_laplacian_variance(camera))
    motor_position.append(motor_y.position)
    diff = laplacian_variance[0] - laplacian_variance[-1]
    print("before diff", laplacian_variance[0], laplacian_variance[-1])
    print("diff", diff)
    if diff > 0:
        print("wrong direction")
        step_size *=-1
        motor_y.move(motor_y.position + 2*step_size, wait=True)
        laplacian_variance.append(calculate_laplacian_variance(camera))
        motor_position.append(motor_y.position)

count = 0
while abs(diff)<tolerance:
    motor_y.move(motor_y.position + step_size, wait=True)
    laplacian_variance.append(calculate_laplacian_variance(camera))
    motor_position.append(motor_y.position)
    print("before diff", laplacian_variance[0], laplacian_variance[-1])
    diff = laplacian_variance[0] - laplacian_variance[-1]
    print("diff", diff)
    count +=1
    if count >5:
        print("couldn't find a best candidate, moving to the best position ")
        print(f"after {len(laplacian_variance)} iterations")
        break

best_position = motor_position[np.argmax(laplacian_variance)]
print("best_position", best_position)
print("laplacian variance", laplacian_variance, len(laplacian_variance))
print("motor_position", motor_position, len(motor_position))
motor_y.move(best_position, wait=True)



            

        


#print(laplacian_var)
#print(motor_y_vals)
#print(max(laplacian_var))
#my_dict = {"motor_y": motor_y_vals, "laplacian_var": laplacian_var, "phi": motor_phi.position,
#    "x": motor_x.position, "y": motor_y.position}

#with open(f'data/{int(motor_phi.position)}_{int(motor_x.position)}_{int(motor_z.position)}.pkl', 'wb') as f:
#    pickle.dump(my_dict, f)