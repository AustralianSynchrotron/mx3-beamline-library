import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class CrystalFinder:
    """
    Calculates the center of mass of individual crystals in a loop, finds the
    size of each crystal in a loop and determines the vertical distance
    between overlapping crystals
    """

    def __init__(self, number_of_spots: npt.NDArray, threshold: float) -> None:
        """
        Parameters
        ----------
        number_of_spots : npt.NDArray
            An array contining the number of spots obtaing from spotfinding.
            The array's shape is (nrows, ncols)
        threshold : float
            Below this threshold, we replace the number with zeros.

        Returns
        -------
        None
        """
        self.number_of_spots = self.filter_array(number_of_spots, threshold)

        self.y_nonzero, self.x_nonzero = np.nonzero(self.number_of_spots)
        self.nonzero_coords = [
            (self.x_nonzero[i], self.y_nonzero[i]) for i in range(len(self.y_nonzero))
        ]

    def center_of_mass(self, number_of_spots) -> tuple[int, int]:
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

    def filter_array(self, array: npt.NDArray, threshold: int) -> npt.NDArray:
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

    def find_adjacent_pixels(self, pixel: tuple[int, int]) -> set[tuple[int, int]]:
        """
        Finds all adjacent pixels of a single pixel containing values different from zero.

        Parameters
        ----------
        pixel : tuple[int, int]
            A pixel coordinate

        Returns
        -------
        set[tuple[int, int]]
            A set containing adjacent pixels of a single pixel
        """

        adjacent_pixels = set()
        for coord in self.nonzero_coords:
            dist = self.distance_between_pixels(pixel, coord)
            if dist <= np.sqrt(2) and coord not in adjacent_pixels:
                adjacent_pixels.update({coord})

        return adjacent_pixels

    def find_individual_islands(
        self, start_coord: tuple[int, int], number_of_spots: npt.NDArray
    ) -> tuple[npt.NDArray, set[tuple[int, int]]]:
        """
        Finds individual islands

        Parameters
        ----------
        start_coord : tuple[int, int]
            Initital coordinate so start finding an island within the
            number of spots array
        number_of_spots : npt.NDArray
            An array containing the number of spots

        Returns
        -------
        tuple[npt.NDArray, set[tuple[int, int]]]
            The indivial island array, and it's corresponding indeces
        """
        island_indeces = set()

        length = [0]
        adjacent_pixels = self.find_adjacent_pixels(start_coord)
        length.append(len(adjacent_pixels))
        island_indeces.update(adjacent_pixels)

        while length[-1] - length[-2]:
            for coord in adjacent_pixels.copy():
                island_indeces.update(self.find_adjacent_pixels(coord))
            adjacent_pixels = island_indeces
            length.append(len(island_indeces))

        island = np.zeros(number_of_spots.shape)
        for index in island_indeces:
            island[index[1]][index[0]] = number_of_spots[index[1]][index[0]]

        return island, island_indeces

    def find_centers_of_mass(self) -> list[tuple[int, int]]:
        """
        Finds the centers of mass of all crystals in the loop

        Returns
        -------
        list[tuple[int, int]]
            A list contining the centers of mass of all crystals in the loop
        """
        list_of_individual_islands = []

        island, island_indeces = self.find_individual_islands(
            (self.x_nonzero[0], self.y_nonzero[0]), self.number_of_spots
        )
        list_of_individual_islands.append(island_indeces)

        island_list_of_arrays = [island]
        for coord in self.nonzero_coords:
            if coord not in island_indeces:
                island_tmp, island_indeces_tmp = self.find_individual_islands(
                    coord, self.number_of_spots
                )
                island_indeces.update(island_indeces_tmp)
                list_of_individual_islands.append(island_indeces_tmp)

                island_list_of_arrays.append(island_tmp)

        center_of_mass_list = []

        for island in island_list_of_arrays:
            center_of_mass_list.append(self.center_of_mass(island))

        return center_of_mass_list

    def distance_between_pixels(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """
        Calculates the distance between two pixels

        Parameters
        ----------
        a : tuple[int, int]
            The coordinate of pixel a
        b : tuple[int, int]
            The coordinate of pixel b

        Returns
        -------
        float
            The distance between pixel a and pixel b
        """
        x = a[0] - b[0]
        y = a[1] - b[1]
        return np.sqrt(x**2 + y**2)

    def plot_centers_of_mass(self, save: bool = False) -> list[tuple[int, int]]:
        """
        Calculates the center of mass of individual crystals in a loop, and plots
        the results.

        Parameters
        ----------
        save : bool, optional
            If true, we save the image

        Returns
        -------
        list[tuple[int, int]]
            A list contining the centers of mass of all crystals in the loop
        """

        center_of_mass_list = self.find_centers_of_mass()

        nx, ny = self.number_of_spots.shape
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)

        X, Y = np.meshgrid(x, y, indexing="ij")

        plt.figure()
        plt.pcolormesh(Y, X, self.number_of_spots, edgecolors="w", cmap="viridis")
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
    # Generate array
    number_of_spots = np.array([100, 100, 0, 100, 120, 100, 100, 100, 0, 100, 100, 0])
    number_of_spots = number_of_spots.reshape(4, 3)
    array_with_zeros = np.append(number_of_spots, np.array([0, 0, 0, 0]))
    rotated_array = np.rot90(
        np.append(array_with_zeros, number_of_spots).reshape(7, 4), k=1
    )
    tmp = np.append(np.zeros((1, rotated_array.shape[1])), rotated_array, axis=0)
    test = np.append(rotated_array, tmp, axis=0)

    # Find centers of mass of the array
    crystal_finder = CrystalFinder(test, threshold=0)
    crystal_finder.plot_centers_of_mass(save=True)
