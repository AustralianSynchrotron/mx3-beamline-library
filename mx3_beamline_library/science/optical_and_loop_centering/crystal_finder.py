import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class CrystalFinder:
    """
    Calculates the center of mass of individual crystals in a loop, finds the
    size of each crystal, and determines the vertical distance
    between overlapping crystals.

    Attributes
    ----------
    y_nonzero : npt.NDArray
        Nonzero y coordinates of self.number_of_spots
    x_nonzero : npt.NDArray
        Nonzero x coordinates of self.number_of_spots
    nonzero_coords : list[tuple[int, int]]
        A list containing nonzero coordinates of the number_of_spots array
    list_of_island_indices : list[set[tuple[int, int]]]
        A list of a set of indices describing individual islands
    list_of_island_arrays : list[npt.NDArray]
        A list of numpy arrays describing individual islands
    """

    def __init__(self, number_of_spots: npt.NDArray, threshold: float) -> None:
        """
        Parameters
        ----------
        number_of_spots : npt.NDArray
            An array containing the number of spots obtained from spotfinding.
            The array's shape should be (n_rows, n_cols)
        threshold : float
            We replace all numbers below this threshold with zeros
            in the number_of_spots array

        Returns
        -------
        None
        """
        self.number_of_spots = self.filter_array(number_of_spots, threshold)

        self.y_nonzero, self.x_nonzero = np.nonzero(self.number_of_spots)
        self.nonzero_coords = [
            (self.x_nonzero[i], self.y_nonzero[i]) for i in range(len(self.y_nonzero))
        ]

        self.list_of_island_indices: list[set[tuple[int, int]]] = None
        self.list_of_island_arrays: list[npt.NDArray] = None

    def center_of_mass(self, number_of_spots: npt.NDArray) -> tuple[int, int]:
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
            The x and y center of mass coordinates which correspond to the x and y indices
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

    def filter_array(self, array: npt.NDArray, threshold: float) -> npt.NDArray:
        """
        Replaces the values of an array with zeros if the array contains numbers below
        a threshold

        Parameters
        ----------
        array : npt.NDArray
            A numpy array containing the number of spots
        threshold : float
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
            The individual island array, and it's corresponding indices
        """
        island_indices = set()

        length = [0]
        adjacent_pixels = self.find_adjacent_pixels(start_coord)
        length.append(len(adjacent_pixels))
        island_indices.update(adjacent_pixels)

        while length[-1] - length[-2]:
            for coord in adjacent_pixels.copy():
                island_indices.update(self.find_adjacent_pixels(coord))
            adjacent_pixels = island_indices
            length.append(len(island_indices))

        island = np.zeros(number_of_spots.shape)
        for index in island_indices:
            island[index[1]][index[0]] = number_of_spots[index[1]][index[0]]

        return island, island_indices

    def find_islands(self) -> None:
        """
        Finds all islands in self.number_of_spots. The values of self.list_of_island_indices and
        self.list_of_island_arrays are updated here.

        Returns
        -------
        None
        """
        self.list_of_island_indices = []

        island, island_indices = self.find_individual_islands(
            (self.x_nonzero[0], self.y_nonzero[0]), self.number_of_spots
        )
        self.list_of_island_indices.append(island_indices.copy())

        self.list_of_island_arrays = [island]
        for coord in self.nonzero_coords:
            if coord not in island_indices:
                island_tmp, island_indices_tmp = self.find_individual_islands(
                    coord, self.number_of_spots
                )
                self.list_of_island_indices.append(island_indices_tmp.copy())
                island_indices.update(island_indices_tmp)
                # self.list_of_island_indices.append(island_indices_tmp.copy())

                self.list_of_island_arrays.append(island_tmp)

    def find_centers_of_mass(self) -> list[tuple[int, int]]:
        """
        Calculates the center of mass of all islands found in self.number_of_spots

        Returns
        -------
        list[tuple[int, int]]
            A list containing the center of mass of individual islands
        """
        if self.list_of_island_arrays is None or self.list_of_island_indices is None:
            self.find_islands()

        center_of_mass_list = []
        for island in self.list_of_island_arrays:
            center_of_mass_list.append(self.center_of_mass(island))

        return center_of_mass_list

    def crystal_locations_and_sizes(self) -> dict:
        """
        Calculates the crystal locations and sizes in a loop. To calculate the
        height and width of the crystal, we assume that the crystal is well
        approximated by a rectangle.

        Returns
        -------
        dict
            A dictionary containing information about the location of the crystal
            as well as its size.
        """
        if self.list_of_island_arrays is None or self.list_of_island_indices is None:
            self.find_islands()

        list_of_crystal_locations_and_sizes = []
        for index in self.list_of_island_indices:
            list_of_crystal_locations_and_sizes.append(self.rectangle_coords(index))
        return list_of_crystal_locations_and_sizes

    def distance_between_overlapping_crystals(self) -> dict[str, int]:
        """
        Calculates the distance between all overlapping crystals in a loop.
        The distances between the ith and jth overlapping crystal is saved in a key
        following the format: f"distance_{i}_{j}"

        Returns
        -------
        dict[str, int]
            A dictionary describing the the distance between all overlapping
            crystals in a loop
        """
        list_of_crystal_locations_and_sizes = self.crystal_locations_and_sizes()
        
        distance_list = []
        for i in range(len(self.list_of_island_indices)):
            for j in range(len(self.list_of_island_indices)):
                if j > i:
                    coords_1 = list_of_crystal_locations_and_sizes[i]
                    coords_2 = list_of_crystal_locations_and_sizes[j]
                    if (
                        coords_2["min_x"] <= coords_1["min_x"] <= coords_2["max_x"]
                    ) or (coords_2["min_x"] <= coords_1["max_x"] <= coords_2["max_x"]):
                        # Note that the -1 is added because we're subtracting indices
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

        return distance_list

    def rectangle_coords(self, island_indices: set[tuple[int, int]]) -> dict:
        """
        Fits a crystal with a rectangle given the indices of an island. Based on that
        assumption we calculate the bottom_left and bottom right coordinates of the
        rectangle, its width, height, and minimum and maximum x and y values

        Parameters
        ----------
        island_indices : set[tuple[int, int]]
            Indices of an island

        Returns
        -------
        dict
            A dictionary containing information about the coordinated of the crystal
            as well as its width and height
        """

        x_vals = []
        y_vals = []
        for coord in island_indices.copy():
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
            "height": height,
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
            A list containing the centers of mass of all crystals in the loop
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
            "X",
            "1",
            "2",
            "<",
            "3",
            "4",
            "^",
            "o",
        ]
        golden_ratio = 1.618
        plt.figure(figsize=[7 * golden_ratio, 7])
        c = plt.pcolormesh(Y, X, self.number_of_spots, edgecolors="w", cmap="viridis")
        for i, center_of_mass in enumerate(center_of_mass_list):
            plt.scatter(
                center_of_mass[0],
                center_of_mass[1],
                label=f"CM: Crystal #{i}",
                marker=marker_list[i],
                s=200,
                color="red",
            )
        plt.colorbar(c)
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

    # Find centers of mass of the array, crystal locations, and distances
    # between overlapping crystals
    crystal_finder = CrystalFinder(test, threshold=0)

    centers_of_mass = crystal_finder.plot_centers_of_mass(save=True)
    crystal_locations_and_sizes = crystal_finder.crystal_locations_and_sizes()
    distance_between_overlapping_crystals = (
        crystal_finder.distance_between_overlapping_crystals()
    )

    print("Centers of mass:\n", centers_of_mass)
    print("\nCrystal locations and sizes:\n", crystal_locations_and_sizes)
    print(
        "\nDistance between overlapping crystals:\n",
        distance_between_overlapping_crystals,
    )
