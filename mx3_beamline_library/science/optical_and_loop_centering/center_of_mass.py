import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def center_of_mass(number_of_spots: npt.NDArray) -> tuple[int, int]:
    """
    Calculate the center of mass of a sample given an array containing
    the number of spots

    Parameters
    ----------
    number_of_spots : npt.NDArray
        An array containing a the number of spots of a sample with shape (n_rows, n_cols)

    Returns
    -------
    x_cm, y_cm : tuple[int, int]
        The x and y center of mass coordinates which correspond to the x and y indeces
        of the numpy array
    """
    shape = number_of_spots.shape

    y_cm = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            y_cm += i * number_of_spots[i][j]
    y_cm = round(y_cm / sum(sum(number_of_spots)))

    x_cm = 0
    for j in range(shape[0]):
        for i in range(shape[1]):
            x_cm += i * number_of_spots[j][i]
    x_cm = round(x_cm / sum(sum(number_of_spots)))

    return x_cm, y_cm


def multi_crystal_center_of_mass(
    number_of_spots: npt.NDArray, threshold: int
) -> list[tuple[int, int]]:
    """
    Calculates the center of mass of individual crystals given an
    array containing the numbers of spots with shape (n_rows, n_cols)

    Parameters
    ----------
    number_of_spots : npt.NDArray
        An array containing the number of spots with shape (n_rows, n_cols)
    threshold : int
        We replace the values of the number_of_spots array below the threshold with zeros

    Returns
    -------
    list[tuple[int,int]]
        A list containing the center of mass of all crystals found in the rastering step
    """
    number_of_spots = filter_array(number_of_spots, threshold)

    # Find the indeces where new crystals are located
    island_start_index = []
    y, x = np.nonzero(number_of_spots)
    for i in range(len(x)):
        if i > 0:
            if (y[i] - y[i - 1]) > 1 or (x[i] - x[i - 1]) > 1:
                island_start_index.append(i)

    # calculate the center of mass of each "island"
    island = np.zeros(np.shape(number_of_spots))
    center_of_mass_list = []
    for i in range(len(x)):
        if i in island_start_index:
            center_of_mass_list.append(center_of_mass(island))
            island = np.zeros(np.shape(number_of_spots))

        island[y[i]][x[i]] = number_of_spots[y[i]][x[i]]

        if i == (len(x) - 1):
            center_of_mass_list.append(center_of_mass(island))

    return center_of_mass_list


def filter_array(array: npt.NDArray, threshold: int) -> npt.NDArray:
    """
    Replaces the values of an array with zeros if the array contains numbers below
    a threshold

    Parameters
    ----------
    array : npt.NDArray
        A numpy array containing the number of spots
    threshold : int
        We replace the values of the array below the threshold with zeros

    Returns
    -------
    npt.NDArray
        The filtered array
    """
    # TODO: there is probably a better way to do this without a for loop
    args = np.argwhere(array < threshold)
    filtered_array = array
    for arg in args:
        filtered_array[tuple(arg)] = 0
    return filtered_array


def plot_muti_crystal_center_of_mass(
    number_of_spots: npt.NDArray, threshold: int, save: bool = False
) -> list[tuple[int, int]]:
    """
    Calculates the center of mass of individual crystals given an
    array containing the numbers of spots with shape (n_rows, n_cols) and plots
    the results.

    Parameters
    ----------
    array : npt.NDArray
        A numpy array containing the number of spots
    threshold : int
        We replace the values of the array below the threshold with zeros
    save : bool, optional
        If true, we save the image
    """

    center_of_mass_list = multi_crystal_center_of_mass(number_of_spots, threshold)

    nx, ny = number_of_spots.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure()
    plt.pcolormesh(Y, X, number_of_spots, edgecolors="w", cmap="viridis")
    for center_of_mass in center_of_mass_list:
        plt.scatter(
            center_of_mass[0],
            center_of_mass[1],
            label="CM",
            marker="+",
            s=200,
            color="red",
        )

    if save:
        plt.savefig("center_of_mass")
    return center_of_mass_list


if __name__ == "__main__":
    number_of_spots = np.array([0, 1, 0, 0, 120, 0, 0, 1, 0])

    number_of_spots = number_of_spots.reshape(3, 3)
    array_with_zeros = np.append(number_of_spots, np.array([0, 0, 0]))
    rotated_array = np.rot90(
        np.append(array_with_zeros, number_of_spots).reshape(7, 3), k=1
    )
    final_test = np.append(rotated_array, rotated_array, axis=0)

    plot_muti_crystal_center_of_mass(final_test, threshold=0, save=True)
