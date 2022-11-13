import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def center_of_mass(number_of_spots: npt.NDArray) -> tuple[int, int]:
    """
    Calculate the center of mass of a sample given an array containing
    the number of spots with shape (n_rows, n_cols)

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

    print("Y axis")
    # Y axis
    center_of_mass_list = find_islands_y_axis(number_of_spots)

    print("------------X axis-------------")
    # X axis
    center_of_mass_list = find_islands_x_axis(number_of_spots)

    # for cm in center_of_mass_rotated:
    #    center_of_mass_list.append(tuple(reversed(cm)))

    print("center of mass:", center_of_mass_list)
    return center_of_mass_list


def find_islands_y_axis(number_of_spots: npt.NDArray):
    y_nonzero, x_nonzero = np.nonzero(number_of_spots)
    print("x", x_nonzero)
    print("y", y_nonzero)
    center_of_mass_list = []

    individual_island = np.zeros(number_of_spots.shape).astype(number_of_spots.dtype)
    # islands accross the y axis
    for i in range(1, len(y_nonzero)):
        if (y_nonzero[i] - y_nonzero[i - 1]) > 1:
            print(f"\nisland {i}\n", individual_island)
            center_of_mass_list.append(center_of_mass(individual_island))
            individual_island = np.zeros(number_of_spots.shape).astype(
                number_of_spots.dtype
            )

        if i == (len(y_nonzero) - 1):
            # print("\n",i)
            print(f"\nisland {i}\n", individual_island)
            center_of_mass_list.append(center_of_mass(individual_island))

        individual_island[y_nonzero[i]][x_nonzero[i]] = number_of_spots[y_nonzero[i]][
            x_nonzero[i]
        ]

    return center_of_mass_list


def find_islands_x_axis(number_of_spots: npt.NDArray):
    number_of_spots = np.rot90(number_of_spots, k=1)

    y_nonzero, x_nonzero = np.nonzero(number_of_spots)
    print("x", x_nonzero)
    print("y", y_nonzero)
    center_of_mass_list = []

    individual_island = np.zeros(number_of_spots.shape).astype(number_of_spots.dtype)
    # islands accross the y axis
    for i in range(1, len(y_nonzero)):
        if (y_nonzero[i] - y_nonzero[i - 1]) > 1:
            print(f"\nisland {i}\n", individual_island)
            center_of_mass_list.append(
                tuple(reversed(center_of_mass(individual_island)))
            )
            individual_island = np.zeros(number_of_spots.shape).astype(
                number_of_spots.dtype
            )

        if i == (len(y_nonzero) - 1):
            # print("\n",i)
            print(f"\nisland {i}\n", individual_island)
            center_of_mass_list.append(
                tuple(reversed(center_of_mass(individual_island)))
            )

        individual_island[y_nonzero[i]][x_nonzero[i]] = number_of_spots[y_nonzero[i]][
            x_nonzero[i]
        ]

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


def find_adjacent_pixels(
    pixel: tuple[int, int], number_of_spots: npt.NDArray
) -> set[tuple[int, int]]:
    y_nonzero, x_nonzero = np.nonzero(number_of_spots)
    # print("original array:\n", number_of_spots)

    nonzero_coords = [(x_nonzero[i], y_nonzero[i]) for i in range(len(y_nonzero))]

    adjacent_pixels = set()
    for coord in nonzero_coords:
        dist = distance_between_pixels(pixel, coord)
        if dist <= np.sqrt(2) and coord not in adjacent_pixels:
            adjacent_pixels.update({coord})

    return adjacent_pixels


def find_individual_islands(start_coord, number_of_spots: npt.NDArray):
    y_nonzero, x_nonzero = np.nonzero(number_of_spots)

    # start_coord = (x_nonzero[0], y_nonzero[0])
    island_indeces = set()

    length = [0]
    adjacent_pixels = find_adjacent_pixels(start_coord, number_of_spots)
    length.append(len(adjacent_pixels))
    island_indeces.update(adjacent_pixels)

    while length[-1] - length[-2]:
        for coord in adjacent_pixels.copy():
            island_indeces.update(find_adjacent_pixels(coord, number_of_spots))
        adjacent_pixels = island_indeces
        length.append(len(island_indeces))
    print(island_indeces)

    island = np.zeros(number_of_spots.shape)
    for index in island_indeces:
        island[index[1]][index[0]] = number_of_spots[index[1]][index[0]]

    print(island)
    return island, island_indeces


def find_all_islands(number_of_spots: npt.NDArray):
    list_of_individual_islands = []

    y_nonzero, x_nonzero = np.nonzero(number_of_spots)
    nonzero_coords = {(x_nonzero[i], y_nonzero[i]) for i in range(len(y_nonzero))}

    island, island_indeces = find_individual_islands(
        (x_nonzero[0], y_nonzero[0]), number_of_spots
    )
    list_of_individual_islands.append(island_indeces)

    island_list_of_arrays = [island]
    for coord in nonzero_coords.copy():
        if coord not in island_indeces:
            island_tmp, island_indeces_tmp = find_individual_islands(
                coord, number_of_spots
            )
            island_indeces.update(island_indeces_tmp)
            list_of_individual_islands.append(island_indeces_tmp)

            island_list_of_arrays.append(island_tmp)

    print(list_of_individual_islands)
    print(island_list_of_arrays)

    # return find_individual_islands((x_nonzero[0], y_nonzero[0]), number_of_spots)


def distance_between_pixels(a: tuple[int, int], b: tuple[int, int]) -> float:
    x = a[0] - b[0]
    y = a[1] - b[1]
    return np.sqrt(x**2 + y**2)


if __name__ == "__main__":
    number_of_spots = np.array([100, 100, 0, 100, 120, 100, 100, 100, 0, 100, 100, 0])

    number_of_spots = number_of_spots.reshape(4, 3)
    print(number_of_spots)
    array_with_zeros = np.append(number_of_spots, np.array([0, 0, 0, 0]))
    rotated_array = np.rot90(
        np.append(array_with_zeros, number_of_spots).reshape(7, 4), k=1
    )
    middle = np.append(np.zeros((1, rotated_array.shape[1])), rotated_array, axis=0)

    final = np.append(rotated_array, middle, axis=0)
    final = np.rot90(np.append(rotated_array, middle, axis=0))

    # plot_muti_crystal_center_of_mass(rotated_array , threshold=0, save=True)

    print(find_all_islands(final))
