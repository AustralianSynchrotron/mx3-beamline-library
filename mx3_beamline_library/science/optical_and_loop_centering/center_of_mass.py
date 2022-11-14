import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class CrystalFinder:
    """
    Calculates the center of mass of individual crystals in a loop, finds the
    size of each crystal, and determines the vertical distance
    between overlapping crystals
    """

    def __init__(self, number_of_spots: npt.NDArray, threshold: float) -> None:
        """
        Parameters
        ----------
        number_of_spots : npt.NDArray
            An array contining the number of spots obtaing from spotfinding.
            The array's shape should be (nrows, ncols)
        threshold : float
            We replace all numbers in the number_of_spots array below this threshold
            with zeros.

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
            Initial coordinate so start finding an island within the
            number of spots array
        number_of_spots : npt.NDArray
            An array containing the number of spots

        Returns
        -------
        tuple[npt.NDArray, set[tuple[int, int]]]
            The individual island array, and it's corresponding indeces
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
        list_of_individual_islands.append(island_indeces.copy())

        island_list_of_arrays = [island]
        for coord in self.nonzero_coords:
            if coord not in island_indeces:
                island_tmp, island_indeces_tmp = self.find_individual_islands(
                    coord, self.number_of_spots
                )
                list_of_individual_islands.append(island_indeces_tmp.copy())
                island_indeces.update(island_indeces_tmp)
                # list_of_individual_islands.append(island_indeces_tmp.copy())

                island_list_of_arrays.append(island_tmp)

        center_of_mass_list = []
        for island in island_list_of_arrays:
            center_of_mass_list.append(self.center_of_mass(island))

        crystal_sizes = []
        for index in list_of_individual_islands:
            size = max(index)[0] - min(index)[0]
            crystal_sizes.append(size)

        distance_list = []
        for i in range(len(list_of_individual_islands)):
            for j in range(len(list_of_individual_islands)):
                if j > i:
                    coords_1 = self.rectangle_coords(list_of_individual_islands[i])
                    coords_2 = self.rectangle_coords(list_of_individual_islands[j])
                    if (
                        coords_2["min_x"] <= coords_1["min_x"] <= coords_2["max_x"]
                    ) or (coords_2["min_x"] <= coords_1["max_x"] <= coords_2["max_x"]):
                        # Note that the -1 is added because we're substracting indeces
                        distance = (
                            min(
                                [
                                    abs(coords_1["max_y"] - coords_2["min_y"]),
                                    abs(coords_2["max_y"] - coords_1["min_y"]),
                                ]
                            )
                            - 1
                        )
                        distance_list.append({f"distance_{i}_{j}": distance})

        print(distance_list)
        return center_of_mass_list

    def rectangle_coords(self, indeces: set[tuple[int, int]]):
        x_vals = []
        y_vals = []
        for coord in indeces.copy():
            x_vals.append(coord[0])
            y_vals.append(coord[1])

        min_x = min(x_vals)
        max_x = max(x_vals)
        min_y = min(y_vals)
        max_y = max(y_vals)

        bottom_left = (min_x, min_y)
        top_right = (max_x, max_y)
        width = max_x - min_x
        height = max_y - min_y
        return {
            "bottom_left": bottom_left,
            "top_right": top_right,
            "width": width,
            "heigth": height,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        }

    def distance_between_pixels(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """
        Calculates the distance between the pixels a and b

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
        marker_list = [
            ".",
            "+",
            "v",
            "p",
            ">",
            "s",
            "P",
            "D",
            "X" "1",
            "2",
            "<",
            "3",
            "4",
            "^",
            "o",
        ]
        plt.figure(figsize=[7 * 1.618, 7])
        plt.pcolormesh(Y, X, self.number_of_spots, edgecolors="w", cmap="viridis")
        for i, center_of_mass in enumerate(center_of_mass_list):
            plt.scatter(
                center_of_mass[0],
                center_of_mass[1],
                label=f"Crystal #{i}",
                marker=marker_list[i],
                s=200,
                color="red",
            )
        plt.legend(labelspacing=1.5)
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
