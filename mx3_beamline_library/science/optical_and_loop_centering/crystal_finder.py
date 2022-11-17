import logging
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import center_of_mass

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


class CrystalFinder:
    """
    Calculates the center of mass of individual crystals in a loop, finds the
    size of each crystal, and determines the vertical distance
    between overlapping crystals.

    Attributes
    ----------
    y_nonzero : npt.NDArray
        Nonzero y coordinates of self.filtered_array
    x_nonzero : npt.NDArray
        Nonzero x coordinates of self.filtered_array
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
            We replace all numbers below this threshold with zeros. The resulting
            arrays is saved as self.filtered_array

        Returns
        -------
        None
        """
        self.filtered_array = np.where(number_of_spots < threshold, 0, number_of_spots)

        self.y_nonzero, self.x_nonzero = np.nonzero(self.filtered_array)
        self.nonzero_coords = [
            (self.x_nonzero[i], self.y_nonzero[i]) for i in range(len(self.y_nonzero))
        ]
        logger.info(f"Number of non-zero pixels: {len(self.nonzero_coords)}")

        self.list_of_island_indices: list[set[tuple[int, int]]] = None
        self.list_of_island_arrays: list[npt.NDArray] = None

    def _find_adjacent_pixels(self, pixel: tuple[int, int]) -> set[tuple[int, int]]:
        """
        Finds all adjacent pixels of a single pixel containing values different from zero.
        Once we have calculated all adjacent pixels, we remove them from self.nonzero_coords
        to avoid counting them again.

        Parameters
        ----------
        pixel : tuple[int, int]
            A pixel coordinate

        Returns
        -------
        set[tuple[int, int]]
            A set containing adjacent pixels of a single pixel
        """
        # Distance between pixels
        diff = np.array(pixel) - np.array(self.nonzero_coords)
        distance_between_pixels = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        adjacent_args = np.argwhere(distance_between_pixels <= np.sqrt(2)).flatten()

        adjacent_pixels = set()
        for arg in adjacent_args:
            adjacent_pixels.update({self.nonzero_coords[arg]})

        # for coord in adjacent_pixels.copy():
        #    self.nonzero_coords.remove(coord)

        return adjacent_pixels

    def _find_individual_islands(
        self, start_coord: tuple[int, int], number_of_spots: npt.NDArray
    ) -> tuple[npt.NDArray, set[tuple[int, int]]]:
        """
        Finds individual islands

        Parameters
        ----------
        start_coord : tuple[int, int]
            We find an island starting from start_coord
        number_of_spots : npt.NDArray
            An array containing the number of spots

        Returns
        -------
        tuple[npt.NDArray, set[tuple[int, int]]]
            The individual island array, and it's corresponding indices
        """
        adjacent_pixels = self._find_adjacent_pixels(start_coord)
        length = [0, len(adjacent_pixels)]
        adjacent_pixels_list = [set(), deepcopy(adjacent_pixels)]

        while length[-1] - length[-2]:
            non_repeated_pixels = adjacent_pixels_list[-1] - adjacent_pixels_list[-2]
            for coord in non_repeated_pixels:
                adjacent_pixels.update(self._find_adjacent_pixels(coord))
            length.append(len(adjacent_pixels))
            adjacent_pixels_list.append(deepcopy(adjacent_pixels))

        island = np.zeros(number_of_spots.shape)
        for index in adjacent_pixels:
            island[index[1]][index[0]] = number_of_spots[index[1]][index[0]]

        return island, adjacent_pixels

    def _find_all_islands(self) -> None:
        """
        Finds all islands in self.filtered_array. The values of self.list_of_island_indices and
        self.list_of_island_arrays are updated here.

        Returns
        -------
        None
        """
        logger.info("Finding islands...")
        t = time.perf_counter()
        self.list_of_island_indices = []

        island, island_indices = self._find_individual_islands(
            (self.x_nonzero[0], self.y_nonzero[0]), self.filtered_array
        )
        self.list_of_island_indices.append(island_indices.copy())

        self.list_of_island_arrays = [island]
        for coord in self.nonzero_coords:
            if coord not in island_indices:
                island_tmp, island_indices_tmp = self._find_individual_islands(
                    coord, self.filtered_array
                )
                self.list_of_island_indices.append(island_indices_tmp.copy())
                island_indices.update(island_indices_tmp)

                self.list_of_island_arrays.append(island_tmp)
        logger.info(
            f"It took {time.perf_counter() - t} [s] to find all islands " "in the loop"
        )

    def find_centers_of_mass(self) -> list[tuple[int, int]]:
        """
        Calculates the center of mass of all islands found in self.filtered_array

        Returns
        -------
        list[tuple[int, int]]
            A list containing the center of mass of individual islands
        """
        if self.list_of_island_arrays is None or self.list_of_island_indices is None:
            self._find_all_islands()

        center_of_mass_list = []
        for island in self.list_of_island_arrays:
            y_cm, x_cm = center_of_mass(island)
            center_of_mass_list.append((round(x_cm), round(y_cm)))

        return center_of_mass_list

    def _crystal_locations_and_sizes(self) -> list[dict]:
        """
        Calculates the crystal locations and sizes in a loop in units of pixels.
        To calculate the height and width of the crystal, we assume that the crystal
        is well approximated by a rectangle.

        Returns
        -------
        list[dict]
            A list of dictionaries containing information about the locations of all
            crystals as well as their sizes.
        """
        if self.list_of_island_arrays is None or self.list_of_island_indices is None:
            self._find_all_islands()

        list_of_crystal_locations_and_sizes = []
        for index in self.list_of_island_indices:
            list_of_crystal_locations_and_sizes.append(self._rectangle_coords(index))
        return list_of_crystal_locations_and_sizes

    def find_crystals_and_overlapping_crystal_distances(
        self,
    ) -> tuple[list[dict], list[dict[str, int]]]:
        """
        Calculates the distance between all overlapping crystals in a loop in units of
        pixels, the crystal locations, and their corresponding sizes (in pixels). The distance
        between the i-th and j-th overlapping crystal is saved in a key following the format:
        f"distance_{i}_{j}"

        Returns
        -------
        list[dict], list[dict[str, int]]
            A list of dictionaries containing information about the locations of all
            crystals as well as their sizes, and a list of dictionaries describing
            the distance between all overlapping crystals in a loop
        """
        list_of_crystal_locations_and_sizes = self._crystal_locations_and_sizes()

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

        return list_of_crystal_locations_and_sizes, distance_list

    def _rectangle_coords(self, island_indices: set[tuple[int, int]]) -> dict:
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

    def plot_crystal_finder_results(
        self,
        save: bool = False,
        interpolation: str = None,
        plot_centers_of_mass: bool = True,
        filename: str = "crystal_finder_results",
    ) -> tuple[list[tuple[int, int]], list[dict], list[dict[str, int]]]:
        """
        Calculates the center of mass of individual crystals in a loop,
        the location and size of all crystals, and estimates
        the distance between overlapping crystals. Finally these results
        are plotted

        Parameters
        ----------
        save : bool, optional
            If true, we save the image
        interpolation : str, optional
            Interpolation used by plt.imshow(). Could be any of the interpolations
            described in the plt.imshow documentation, by default None
        plot_center_of_mass : bool, optional
            If true, we plot the centers of mass
        filename : str
            Name of the image. The filename is used only if save=True,
            by default crystal_finder_results

        Returns
        -------
        tuple[list[tuple[int, int]], list[dict], list[dict[str, int]]]
            A list containing the centers of mass of all crystals in the loop,
            a list of dictionaries containing information about the locations ans sizes
            of all crystals, and a list of dictionaries describing the distance between
            all overlapping crystals in a loop
        """

        center_of_mass_list = self.find_centers_of_mass()
        (
            list_of_crystal_locations_and_sizes,
            distance_list,
        ) = self.find_crystals_and_overlapping_crystal_distances()

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
        c = plt.imshow(self.filtered_array, interpolation=interpolation)
        if plot_centers_of_mass:
            for i, _center_of_mass in enumerate(center_of_mass_list):
                try:
                    plt.scatter(
                        _center_of_mass[0],
                        _center_of_mass[1],
                        label=f"CM: Crystal #{i}",
                        marker=marker_list[i],
                        s=200,
                        color="red",
                    )
                except IndexError:  # we ran out of markers :/
                    plt.scatter(
                        _center_of_mass[0],
                        _center_of_mass[1],
                        label=f"CM: Crystal #{i}",
                        s=200,
                        color="red",
                    )
            plt.legend(labelspacing=1.5)

        for crystal_locations in list_of_crystal_locations_and_sizes:
            self._plot_rectangle_surrounding_crystal(crystal_locations)

        plt.colorbar(c, label="Number of spots")
        if save:
            plt.savefig(filename)
        return center_of_mass_list, list_of_crystal_locations_and_sizes, distance_list

    def _plot_rectangle_surrounding_crystal(
        self,
        rectangle_coordinates: dict,
    ) -> None:
        """
        Plots a rectangle surrounding a crystal

        Parameters
        ----------
        rectangle_coordinates : dict
            A dictionary obtained from the self._rectangle_coords method

        Returns
        -------
        None
        """
        # top
        x = np.linspace(
            rectangle_coordinates["bottom_left"][0] - 0.5,
            rectangle_coordinates["top_right"][0] + 0.5,
            100,
        )
        z = (rectangle_coordinates["bottom_left"][1] - 0.5) * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Bottom
        x = np.linspace(
            rectangle_coordinates["bottom_left"][0] - 0.5,
            rectangle_coordinates["top_right"][0] + 0.5,
            100,
        )
        z = (rectangle_coordinates["top_right"][1] + 0.5) * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Right side
        z = np.linspace(
            rectangle_coordinates["bottom_left"][1] - 0.5,
            rectangle_coordinates["top_right"][1] + 0.5,
            100,
        )
        x = (rectangle_coordinates["top_right"][0] + 0.5) * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Left side
        z = np.linspace(
            rectangle_coordinates["bottom_left"][1] - 0.5,
            rectangle_coordinates["top_right"][1] + 0.5,
            100,
        )
        x = (rectangle_coordinates["bottom_left"][0] - 0.5) * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")


if __name__ == "__main__":
    import cv2

    img = cv2.imread(
        "/mnt/shares/smd_share/crystal_finder_test_images/crystal_10.tif", 0
    )
    img = img.__invert__()
    img = img[::2, ::2]  # Downsample img by a factor of 2

    # Find centers of mass of the array, crystal locations, and distances
    # between overlapping crystals
    t = time.perf_counter()
    crystal_finder = CrystalFinder(img, threshold=100)

    (
        centers_of_mass,
        locations_and_sizes,
        distances,
    ) = crystal_finder.plot_crystal_finder_results(save=True)

    print("Centers of mass:\n", centers_of_mass)
    print("\nCrystal locations and sizes:\n", locations_and_sizes)
    print(
        "\nDistance between overlapping crystals:\n",
        distances,
    )

    print("Total processing time: ", time.perf_counter() - t)
