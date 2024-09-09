import logging
import os
from datetime import datetime
from os import environ
from typing import Callable, Generator, Literal, Union
from warnings import warn

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluesky.plan_stubs import move_per_step, mv, trigger_and_read
from bluesky.plans import grid_scan, scan
from bluesky.utils import Msg
from dateutil import tz
from ophyd.areadetector.base import EpicsSignalWithRBV
from ophyd.epics_motor import EpicsMotor
from scipy.constants import golden_ratio
from scipy.stats import skewnorm

from ...devices.classes.detectors import BlackflyCamera, HDF5Filewriter
from .stats import Scan1DStats

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)

HDF5_OUTPUT_DIRECTORY = environ.get("HDF5_OUTPUT_DIRECTORY", os.getcwd())


class Scan1D:
    """
    This class is used to run a 1D scan. The resulting distribution is fitted
    using the scipy.stats.skewnorm model
    """

    def __init__(
        self,
        detectors: list[BlackflyCamera | HDF5Filewriter | EpicsSignalWithRBV],
        motor: EpicsMotor,
        initial_position: float,
        final_position: float,
        number_of_steps: int,
        calculate_first_derivative: bool = False,
        dwell_time: float = 0,
        metadata: dict = None,
        hdf5_filename: str = None,
        stop_plan_criteria: (
            Literal["gaussian"] | Callable[[list[float]], bool] | None
        ) = None,
    ) -> None:
        """
        Parameters
        ----------
        detectors : list[BlackflyCamera | HDF5Filewriter | EpicsSignalWithRBV]
            A list of detectors. By default, statistics will be calculated on the first detector
            e.g. if detectors=[my_camera.stats.total, my_camera.stats.sigma],
            statistics will be calculated on my_camera.stats.total
        motor : EpicsMotor
            The motor being scanned.
        initial_position : float
            The initial position of the motor.
        final_position : float
            The final position of the motor.
        number_of_steps : int
            The number of steps in the scan.
        calculate_first_derivate: bool, optional
            If True, we calculate the first derivative of the data generated during
            the scan. The distribution of the first derivative of the data is assumed
            to be Gaussian.
        dwell_time: float, optional
            Amount of time to wait after moves to report status completion, by default 0
        metadata : dict, optional
            Additional metadata associated with the scan, by default None
        hdf5_filename : str, optional
            Name of the generated HDF5 file. If not provided, the filename is based
            on the generation time.
            Example: "mx3_1d_scan_28-08-2023_05:44:15.h5"
        stop_plan_criteria: Literal["gaussian"] | Callable[[list], bool] | None
            The stop plan criteria. Defaults to None (no stop criteria).
            - "gaussian": Stops when the full Gaussian distribution is sampled.
            - A custom function with signature Callable[[list[float]], bool] can be passed.
                The function takes a list of floats and returns True to stop the plan.
            An example of such a function is:

            def stop_criteria(stats_list: list) -> bool:
                return False
        """

        self.motor = motor
        self.detectors = detectors
        self.initial_position = initial_position
        self.final_position = final_position
        self.number_of_steps = number_of_steps
        self.metadata = metadata
        self.hdf5_filename = hdf5_filename
        self.calculate_first_derivative = calculate_first_derivative
        self.dwell_time = dwell_time
        self.motor.settle_time = dwell_time

        self.motor_array = np.linspace(
            initial_position, final_position, number_of_steps
        )
        self.intensity_dict = None
        self.first_derivative = None
        self._stop_plan: bool = False

        self.detector_names = None

        self.stop_plan_criteria = stop_plan_criteria

        _check_motor_limits(
            motor=self.motor,
            min_value=self.initial_position,
            max_value=self.final_position,
        )

    def _check_if_master_file_exists(self) -> None:
        """Checks if the master file exists

        Raises
        ------
        FileExistsError
            Raises and error if the master file exists
        """
        if self.hdf5_filename is not None:
            _hdf5_path = os.path.join(HDF5_OUTPUT_DIRECTORY, self.hdf5_filename)
            name, file_extension = os.path.splitext(_hdf5_path)

            if file_extension != ".h5":
                logger.warning(
                    "HDF5 filename extension does not end with `.h5`. File will "
                    f"be renamed to : {name + '.h5'}"
                )
                name = name + ".h5"

            self.hdf5_filename = os.path.join(HDF5_OUTPUT_DIRECTORY, name)
            if os.path.isfile(self.hdf5_filename):
                raise FileExistsError(
                    f"{self.hdf5_filename} already exists. Choose a different file name"
                )
            return
        else:
            now = datetime.now(tz=tz.gettz("Australia/Melbourne"))
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            name = "mx3_1d_scan_" + dt_string + ".h5"

            self.hdf5_filename = os.path.join(HDF5_OUTPUT_DIRECTORY, name)

    def run(self) -> Generator[Msg, None, None]:
        """
        This function runs a 1D scan until the scanned distribution is within
        one-sigma of the maximum value of the Gaussian.
        NOTE: The stop criteria for the scan is still under development
        and subject to change.


        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan for the scan.

        Raises
        ------
        ValueError
            If metadata is not a dictionary.
        """
        self._check_if_master_file_exists()

        if self.metadata is None:
            self.metadata = {"hdf5_filename": self.hdf5_filename}
        elif isinstance(self.metadata, dict):
            self.metadata.update({"hdf5_filename": self.hdf5_filename})
        elif not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        frame_filewriter_signal_index = None
        for index, detector in enumerate(self.detectors):
            detector.kind = "hinted"

            if isinstance(detector, HDF5Filewriter):
                detector: HDF5Filewriter
                frame_filewriter_signal_index = index
                yield from mv(
                    detector.filename,
                    self.hdf5_filename,
                    detector.frames_per_datafile,
                    self.number_of_steps,
                )
                write_path_template = detector.write_path_template.get()
                self.metadata.update({"write_path_template": write_path_template})

        self.metadata.update({"favourite": False, "favourite_description": ""})
        self._stats_buffer = None
        self.detector_names = None
        self.statistics = []
        yield from scan(
            self.detectors,
            self.motor,
            self.initial_position,
            self.final_position,
            self.number_of_steps,
            num=None,
            per_step=self._one_nd_step,
            md=self.metadata,
        )
        self.intensity_dict: dict = self._stats_buffer

        for key, intensity_array in self.intensity_dict.items():
            self.updated_motor_positions = self.motor_array[: len(intensity_array)]
            if self.calculate_first_derivative:
                self.first_derivative = np.gradient(intensity_array)
                if intensity_array[0] > intensity_array[-1]:
                    self._flipped_gaussian = True
                else:
                    self._flipped_gaussian = False
                self.stats_class = Scan1DStats(
                    self.updated_motor_positions,
                    self.first_derivative,
                    self._flipped_gaussian,
                )
            else:
                self.stats_class = Scan1DStats(
                    self.updated_motor_positions, intensity_array
                )
            self._plot_results(
                self.updated_motor_positions, intensity_array, self.stats_class, key
            )

            self.data = pd.DataFrame(
                {
                    "motor_positions": self.updated_motor_positions,
                    "intensity": intensity_array,
                }
            )
            # TODO: FIX filewriter

        if frame_filewriter_signal_index is not None:
            self._add_metadata_to_hdf5_file(
                self.detectors[frame_filewriter_signal_index]
            )
        else:
            logger.warning(
                "A HDF5Filewriter signal has not been specified in the detector list. "
                "Frames will not be added to the HDF5 file."
            )
            self._add_metadata_to_hdf5_file()

    def _add_metadata_to_hdf5_file(
        self, hdf5_filewriter_signal: HDF5Filewriter | None = None
    ) -> None:
        """Adds metadata to hdf5 files

        Parameters
        ----------
        hdf5_filewriter_signal : HDF5Filewriter | None
            A HDF5Filewriter object

        Returns
        -------
        None
        """
        detector_str = []
        for det in self.detectors:
            detector_str.append(det.__str__())

        if hdf5_filewriter_signal is not None:
            mode = "r+"
        else:
            mode = "w"

        with h5py.File(self.hdf5_filename, mode) as f:
            f.create_dataset(
                "/entry/data/motor_positions",
                data=np.array(self.updated_motor_positions),
            )

            for key, intensity_array in self.intensity_dict.items():
                f.create_dataset(
                    f"/entry/data/{key}",
                    data=np.array(intensity_array),
                )
            f.create_dataset(
                "entry/scan_parameters/motor",
                data=self.motor.__str__(),
            )
            f.create_dataset(
                "entry/scan_parameters/detectors",
                data=detector_str,
            )
            f.create_dataset(
                "entry/scan_parameters/initial_position",
                data=self.initial_position,
            )
            f.create_dataset(
                "entry/scan_parameters/final_position",
                data=self.final_position,
            )
            f.create_dataset(
                "entry/scan_parameters/number_of_steps",
                data=self.number_of_steps,
            )
            f.create_dataset(
                "entry/scan_parameters/dwell_time",
                data=self.dwell_time,
            )
            f.create_dataset(
                "entry/scan_parameters/calculate_first_derivative",
                data=self.calculate_first_derivative,
            )

    def _one_nd_step(
        self,
        detectors: list,
        step: dict,
        pos_cache: dict,
        take_reading: Callable = trigger_and_read,
    ) -> Generator[Msg, None, None]:
        """
        Inner loop of the mx3_1d_scan

        Parameters
        ----------
        detectors : list
            devices to read
        step : dict
            mapping motors to positions in this step
        pos_cache : dict
            mapping motors to their last-set positions
        take_reading : Callable, optional
            function to do the actual acquisition ::

            Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

            Defaults to `trigger_and_read`

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan for the scan.
        """

        if not self._stop_plan:
            motors = step.keys()
            yield from move_per_step(step, pos_cache)
            reading: dict = yield from take_reading(list(motors) + list(detectors))

            if self.detector_names is None:
                _det_names = set(d.name for d in list(detectors))
                _reading_keys = set(reading.keys())
                self.detector_names = list(_reading_keys & _det_names)
                self._stats_buffer = dict()
                for det in self.detector_names:
                    self._stats_buffer[det] = []

            for det in self.detector_names:
                self._stats_buffer[det].append(reading[det]["value"])

            if self.stop_plan_criteria is None:
                return

            if callable(self.stop_plan_criteria):
                self._stop_plan = self.stop_plan_criteria(self._stats_buffer)
                return

            if self.stop_plan_criteria.lower() == "gaussian":
                self._stop_plan = self._gaussian_stop_plan_criteria(self._stats_buffer)
            else:
                raise NotImplementedError(
                    "Stop plan criteria can be None, gaussian, or a "
                    "callable function with signature Callable[[list], bool], "
                    f"not {self.stop_plan_criteria}"
                )

    def _gaussian_stop_plan_criteria(self, stats_list: list) -> bool:
        """
        Criteria used to determine when to stop the mx3_1d_scan plan.

        Parameters
        ----------
        stats_list : list
            A list of numerical values representing relevant statistics.

        Returns
        -------
        bool
            Returns True if the stop criteria have been met; otherwise, returns False.
        """
        if len(stats_list) < 5:
            return False
        else:
            argmax = np.argmax(stats_list)
            if len(stats_list) == argmax * 2:
                logger.info("Stop criteria satisfied. Scan will stop now.")
                return True
            else:
                return False

    def _plot_results(
        self, motor_positions, intensity_array, stats: Scan1DStats, detector_name
    ) -> None:
        """
        Plots the fitted curve along with the raw data and the fit parameters

        Returns
        -------
        None
        """
        if self.calculate_first_derivative:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[2 * 4 * golden_ratio, 4])
            ax[0].scatter(motor_positions, intensity_array, label="Raw data")
            ax[0].set_xlabel(self.motor.name)
            ax[0].set_ylabel(detector_name)
            ax[0].legend()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[4 * golden_ratio, 4])
            ax = [None, ax]

        if self.calculate_first_derivative:
            ax[1].scatter(
                motor_positions,
                self.first_derivative,
                label=r"$\frac{df}{dx}$",
            )
            ax[1].set_xlabel(self.motor.name)
            ax[1].set_ylabel(r"$\frac{df}{dx}$")
        else:
            ax[1].scatter(
                motor_positions,
                intensity_array,
                label=r"Raw Data",
            )
            ax[1].set_xlabel(self.motor.name)
            ax[1].set_ylabel(detector_name)

        # Prepare fitted plots
        statistics = stats.calculate_stats()
        self.statistics.append(statistics)
        if statistics is None:
            ax[1].legend()
            plt.tight_layout()
            plt.show()
            return

        if statistics.FWHM is None or not min(
            motor_positions
        ) <= statistics.mean <= max(motor_positions):
            warn(
                "Gaussian fit may not be accurate, "
                f"statistics for {detector_name} will not be plotted. "
                f"Covariance matrix: {statistics.skewnorm_fit_parameters.covariance_matrix}"
            )
            ax[1].legend()
            plt.tight_layout()
            plt.show()
            return
        x_tmp = np.linspace(min(motor_positions), max(motor_positions), 4096)
        fitted_func = (
            statistics.skewnorm_fit_parameters.pdf_scaling_constant
            * skewnorm.pdf(
                x_tmp,
                statistics.skewnorm_fit_parameters.a,
                statistics.skewnorm_fit_parameters.location,
                statistics.skewnorm_fit_parameters.scale,
            )
            + statistics.skewnorm_fit_parameters.offset
        )
        y_tmp = fitted_func * stats.normalisation_constant
        mean = round(statistics.mean, 2)
        peak = (round(statistics.peak[0], 2), round(statistics.peak[1], 2))
        sigma = round(statistics.sigma, 2)
        skewness = round(statistics.skewness, 2)
        label = (
            r"$\bf{"
            + "Curve"
            + "}$"
            + r" $\bf{"
            + "Fit:"
            + "}$"
            + f"\n$\sigma={sigma}$  \nSkewness={skewness} "
        )
        if self.calculate_first_derivative:
            if self._flipped_gaussian:
                ax[1].plot(
                    x_tmp, -1 * y_tmp, label=label, linestyle="--", color="tab:orange"
                )
            else:
                ax[1].plot(
                    x_tmp, y_tmp, label=label, linestyle="--", color="tab:orange"
                )
        else:
            ax[1].plot(x_tmp, y_tmp, label=label, linestyle="--", color="tab:orange")
        ax[1].axvline(
            statistics.mean, linestyle="--", label=f"$\mu={mean}$", color="gray"
        )
        ax[1].scatter(
            statistics.peak[0],
            statistics.peak[1],
            label=f"Peak={peak}",
            marker="2",
            s=300,
            color="k",
        )
        if statistics.FWHM is not None:
            FWHM = round(statistics.FWHM, 2)
            plt.axvspan(
                xmin=statistics.FWHM_x_coords[0],
                xmax=statistics.FWHM_x_coords[1],
                alpha=0.2,
                label=f"\nFWHM={FWHM}",
            )
        ax[1].legend()
        plt.tight_layout()
        # plt.savefig(f"stats_{detector_name}")
        plt.show()


class Scan2D:
    """
    This class is used to run a 2D grid scan
    """

    def __init__(
        self,
        detectors: list[Union[BlackflyCamera, HDF5Filewriter, EpicsSignalWithRBV]],
        motor_1: EpicsMotor,
        initial_position_motor_1: float,
        final_position_motor_1: float,
        number_of_steps_motor_1: int,
        motor_2: EpicsMotor,
        initial_position_motor_2: float,
        final_position_motor_2: float,
        number_of_steps_motor_2: int,
        dwell_time: float = 0,
        metadata: dict = None,
        hdf5_filename: str = None,
    ) -> None:
        """
        Parameters
        ----------
        detectors : list[Union[BlackflyCamera, HDF5Filewriter, EpicsSignalWithRBV]]
            A list of detectors, e.g., [my_camera.stats.total, my_camera.stats.sigma].
        motor : EpicsMotor
            The motor being scanned.
        initial_position : float
            The initial position of the motor.
        final_position : float
            The final position of the motor.
        number_of_steps : int
            The number of steps in the scan.
        calculate_first_derivate: bool, optional
            If True, we calculate the first derivative of the data generated during
            the scan. The distribution of the first derivative of the data is assumed
            to be Gaussian.
        dwell_time: float, optional
            Amount of time to wait after moves to report status completion, by default 0
        metadata : dict, optional
            Additional metadata associated with the scan, by default None
        hdf5_filename : str, optional
            Name of the generated HDF5 file. If not provided, the filename is based
            on the generation time.
            Example: "mx3_1d_scan_28-08-2023_05:44:15.h5"
        """
        self.detectors = detectors
        self.motor_1 = motor_1
        self.initial_position_motor_1 = initial_position_motor_1
        self.final_position_motor_1 = final_position_motor_1
        self.number_of_steps_motor_1 = number_of_steps_motor_1
        self.motor_2 = motor_2
        self.initial_position_motor_2 = initial_position_motor_2
        self.final_position_motor_2 = final_position_motor_2
        self.number_of_steps_motor_2 = number_of_steps_motor_2
        self.metadata = metadata
        self.hdf5_filename = hdf5_filename

        self.dwell_time = dwell_time
        self.motor_1.settle_time = dwell_time
        self.motor_2.settle_time = dwell_time

        _check_motor_limits(
            motor=self.motor_1,
            min_value=self.initial_position_motor_1,
            max_value=self.final_position_motor_1,
        )
        _check_motor_limits(
            motor=self.motor_2,
            min_value=self.initial_position_motor_2,
            max_value=self.final_position_motor_2,
        )

    def _check_if_master_file_exists(self) -> None:
        """Checks if the master file exists

        Raises
        ------
        FileExistsError
            Raises and error if the master file exists
        """
        if self.hdf5_filename is not None:
            _hdf5_path = os.path.join(HDF5_OUTPUT_DIRECTORY, self.hdf5_filename)
            name, file_extension = os.path.splitext(_hdf5_path)

            if file_extension != ".h5":
                logger.warning(
                    "HDF5 filename extension does not end with `.h5`. File will "
                    f"be renamed to : {name + '.h5'}"
                )
                name = name + ".h5"

            self.hdf5_filename = os.path.join(HDF5_OUTPUT_DIRECTORY, name)
            if os.path.isfile(self.hdf5_filename):
                raise FileExistsError(
                    f"{self.hdf5_filename} already exists. Choose a different file name"
                )
            return
        else:
            now = datetime.now(tz=tz.gettz("Australia/Melbourne"))
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            name = "mx3_2d_scan_" + dt_string + ".h5"

            self.hdf5_filename = os.path.join(HDF5_OUTPUT_DIRECTORY, name)

    def run(self) -> Generator[Msg, None, None]:
        """
        This function runs a 2D grid scan. The frames generated during each run is saved
        to HDF5 files and metadata can be saved to MongoDB using tiled

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan for the scan.

        Raises
        ------
        ValueError
            If metadata is not a dictionary.
        """
        self._check_if_master_file_exists()

        if self.metadata is None:
            self.metadata = {"hdf5_filename": self.hdf5_filename}
        elif isinstance(self.metadata, dict):
            self.metadata.update({"hdf5_filename": self.hdf5_filename})
        elif not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")

        frame_filewriter_signal_index = None
        for index, detector in enumerate(self.detectors):
            detector.kind = "hinted"

            if isinstance(detector, HDF5Filewriter):
                detector: HDF5Filewriter
                frame_filewriter_signal_index = index
                yield from mv(
                    detector.filename,
                    self.hdf5_filename,
                    detector.frames_per_datafile,
                    self.number_of_steps_motor_1 * self.number_of_steps_motor_2,
                )
                write_path_template = detector.write_path_template.get()
                self.metadata.update({"write_path_template": write_path_template})

        self.metadata.update({"favourite": False, "favourite_description": ""})
        self.intensity = None
        self.detector_names = None
        yield from grid_scan(
            self.detectors,
            self.motor_1,
            self.initial_position_motor_1,
            self.final_position_motor_1,
            self.number_of_steps_motor_1,
            self.motor_2,
            self.initial_position_motor_2,
            self.final_position_motor_2,
            self.number_of_steps_motor_2,
            per_step=self._one_nd_step,
            md=self.metadata,
        )

        for key, intensity in self.intensity.items():
            self._plot_heatmap(intensity, key)

        if frame_filewriter_signal_index is not None:
            self._add_metadata_to_hdf5_file(
                self.detectors[frame_filewriter_signal_index]
            )
        else:
            logger.warning(
                "A HDF5Filewriter signal has not been specified in the detector list. "
                "Frames will not be added to the HDF5 file."
            )
            self._add_metadata_to_hdf5_file()

    def _add_metadata_to_hdf5_file(
        self, hdf5_filewriter_signal: HDF5Filewriter | None = None
    ) -> None:
        """Adds metadata to hdf5 files

        Parameters
        ----------
        hdf5_filewriter_signal : HDF5Filewriter | None, optional
            A HDF5Filewriter object, by default None

        Returns
        -------
        None
        """
        detector_str = []
        for det in self.detectors:
            detector_str.append(det.__str__())

        if hdf5_filewriter_signal is not None:
            mode = "r+"

        else:
            mode = "w"

        with h5py.File(self.hdf5_filename, mode=mode) as f:

            for key, intensity in self.intensity.items():
                intensity_dataset = f.create_dataset(
                    f"/entry/data/{key}",
                    data=np.array(intensity).reshape(
                        self.number_of_steps_motor_1, self.number_of_steps_motor_2
                    ),
                )
                intensity_dataset.attrs["metadata"] = (
                    "x axis corresponds to motor_1 and y axis corresponds to motor_2"
                )
            f.create_dataset(
                "entry/scan_parameters/motor_1",
                data=self.motor_1.__str__(),
            )
            f.create_dataset(
                "entry/scan_parameters/motor_2",
                data=self.motor_2.__str__(),
            )
            f.create_dataset(
                "entry/scan_parameters/detectors",
                data=detector_str,
            )
            f.create_dataset(
                "entry/scan_parameters/initial_position_motor_1",
                data=self.initial_position_motor_1,
            )
            f.create_dataset(
                "entry/scan_parameters/final_position_motor_1",
                data=self.final_position_motor_1,
            )
            f.create_dataset(
                "entry/scan_parameters/number_of_steps_motor_1",
                data=self.number_of_steps_motor_1,
            )
            f.create_dataset(
                "entry/scan_parameters/initial_position_motor_2",
                data=self.initial_position_motor_2,
            )
            f.create_dataset(
                "entry/scan_parameters/final_position_motor_2",
                data=self.final_position_motor_2,
            )
            f.create_dataset(
                "entry/scan_parameters/number_of_steps_motor_2",
                data=self.number_of_steps_motor_2,
            )
            f.create_dataset(
                "entry/scan_parameters/dwell_time",
                data=self.dwell_time,
            )

    def _one_nd_step(
        self,
        detectors: list,
        step: dict,
        pos_cache: dict,
        take_reading: Callable = trigger_and_read,
    ) -> Generator[Msg, None, None]:
        """
        Inner loop of the mx3_1d_scan

        Parameters
        ----------
        detectors : list
            devices to read
        step : dict
            mapping motors to positions in this step
        pos_cache : dict
            mapping motors to their last-set positions
        take_reading : Callable, optional
            function to do the actual acquisition ::

            Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

            Defaults to `trigger_and_read`

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan for the scan.
        """

        motors = step.keys()
        yield from move_per_step(step, pos_cache)
        reading = yield from take_reading(list(detectors) + list(motors))

        if self.detector_names is None:
            _det_names = set(d.name for d in list(detectors))
            _reading_keys = set(reading.keys())
            self.detector_names = list(_reading_keys & _det_names)
            self.intensity = dict()
            for det in self.detector_names:
                self.intensity[det] = []

        for det in self.detector_names:
            self.intensity[det].append(reading[det]["value"])

    def _plot_heatmap(self, intensity: list, detector_name: str) -> None:
        """
        Plots a heatmap

        Parameters
        ----------
        intensity : list
            The intensity list
        detector_name : str
            The detector name

        Returns
        -------
        None
        """
        plt.figure(figsize=[4 * golden_ratio, 4])

        plt.imshow(
            np.array(intensity).reshape(
                self.number_of_steps_motor_1, self.number_of_steps_motor_2
            ),
            extent=[
                self.initial_position_motor_2,
                self.final_position_motor_2,
                self.initial_position_motor_1,
                self.final_position_motor_1,
            ],
            origin="lower",
        )
        plt.xlabel(self.motor_2.name)
        plt.ylabel(self.motor_1.name)
        plt.colorbar(label=detector_name)
        plt.show()


def _check_motor_limits(motor: EpicsMotor, min_value: float, max_value: float) -> None:
    """
    Checks if the user input values are within the motor limits

    Parameters
    ----------
    motor : EpicsMotor
        The motor class
    min_value : float
        The minimum value
    max_value : float
        The maximum value

    Raises
    ------
    ValueError
        Raises an error if the motor positions are outside of the limits
    """
    try:
        limits = motor.limits
    except AttributeError:
        warn(f"Motor {motor.name} does not have limits. Limits will not be checked")
        return

    try:
        if not isinstance(limits[0], float) or not isinstance(limits[1], float):
            warn(f"Motor {motor.name} does not have limits. Limits will not be checked")
            return
    except Exception:
        warn(f"Motor {motor.name} does not have limits. Limits will not be checked")
        return

    if min_value < limits[0]:
        raise ValueError(
            f"Minimum limit exceeded for motor {motor.name}. Requested value was "
            f"{min_value}, but the lower limit is {limits[0]}"
        )
    if max_value > limits[1]:
        raise ValueError(
            f"Maximum limit exceeded for motor {motor.name}. Requested value was "
            f"{max_value}, but the upper limit is {limits[1]}"
        )
