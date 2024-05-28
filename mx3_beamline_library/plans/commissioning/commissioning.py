import logging
from datetime import datetime, timezone
from typing import Callable, Generator, Union

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


class Scan1D:
    """
    This class is used to run a 1D scan. The resulting distribution is fitted
    using the scipy.stats.skewnorm model
    """

    def __init__(
        self,
        detectors: list[Union[BlackflyCamera, HDF5Filewriter, EpicsSignalWithRBV]],
        motor: EpicsMotor,
        initial_position: float,
        final_position: float,
        number_of_steps: int,
        calculate_first_derivative: bool = False,
        dwell_time: float = 0,
        metadata: dict = None,
        hdf5_filename: str = None,
        calculate_stats_from_detector_name: str = None,
    ) -> None:
        """
        Parameters
        ----------
        detectors : list[Union[BlackflyCamera, HDF5Filewriter, EpicsSignalWithRBV]]
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
        calculate_stats_from_detector_name : str, optional
            The name of the detector for which statistics should be calculated.
            If not specified, statistics will be calculated for the first detector
            in the detector list by default.
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
        self.intensity_array = None
        self.first_derivative = None
        self._stop_plan: bool = False
        self._filewriter_mode = False

        if calculate_stats_from_detector_name is None:
            self.calculate_stats_from_detector_name = detectors[0].name
        else:
            self.calculate_stats_from_detector_name = calculate_stats_from_detector_name

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

        if self.hdf5_filename is None:
            now = datetime.now(tz=tz.gettz("Australia/Melbourne"))
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            self.hdf5_filename = "mx3_1d_scan_" + dt_string + ".h5"

        if self.metadata is None:
            self.metadata = {"hdf5_filename": self.hdf5_filename}
        elif type(self.metadata) is dict:
            self.metadata.update({"hdf5_filename": self.hdf5_filename})
        elif type(self.metadata) is not dict:
            raise ValueError("Metadata must be a dictionary")

        for index, detector in enumerate(self.detectors):
            detector.kind = "hinted"

            if type(detector) == HDF5Filewriter:
                self._filewriter_mode = True
                detector: HDF5Filewriter
                filewriter_signal_index = index
                yield from mv(
                    detector.filename,
                    self.hdf5_filename,
                    detector.frames_per_datafile,
                    self.number_of_steps,
                )
                write_path_template = detector.write_path_template.get()
                self.metadata.update({"write_path_template": write_path_template})

        if not self._filewriter_mode:
            logger.warning(
                "A HDF5Filewriter signal has not been specified in the detector list. "
                "HDF5 files will not be created"
            )

        self.metadata.update({"favourite": False, "favourite_description": ""})
        self._stats_buffer = []
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
        self.intensity_array = self._stats_buffer
        self.updated_motor_positions = self.motor_array[: len(self.intensity_array)]

        if len(self.intensity_array) > 4:
            if self.calculate_first_derivative:
                self.first_derivative = np.gradient(self.intensity_array)
                if self.intensity_array[0] > self.intensity_array[-1]:
                    self._flipped_gaussian = True
                else:
                    self._flipped_gaussian = False
                stats = Scan1DStats(
                    self.updated_motor_positions,
                    self.first_derivative,
                    self._flipped_gaussian,
                )
                self.statistics = stats.calculate_stats()
            else:
                stats = Scan1DStats(self.updated_motor_positions, self.intensity_array)
                self.statistics = stats.calculate_stats()
            self._plot_results()
        else:
            logger.info(
                "Statistics could not be calculated, at least 5 motor positions are needed"
            )

        self.data = pd.DataFrame(
            {
                "motor_positions": self.updated_motor_positions,
                "intensity": self.intensity_array,
            }
        )
        if self._filewriter_mode:
            self._add_metadata_to_hdf5_file(self.detectors[filewriter_signal_index])

    def _add_metadata_to_hdf5_file(
        self, hdf5_filewriter_signal: HDF5Filewriter
    ) -> None:
        """Adds metadata to hdf5 files

        Parameters
        ----------
        hdf5_filewriter_signal : HDF5Filewriter
            A HDF5Filewriter object

        Returns
        -------
        None
        """
        detector_str = []
        for det in self.detectors:
            detector_str.append(det.__str__())

        with h5py.File(hdf5_filewriter_signal.hdf5_path, mode="r+") as f:
            f.create_dataset(
                "/entry/data/motor_positions_vs_intensity",
                data=np.transpose([self.updated_motor_positions, self.intensity_array]),
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
            reading = yield from take_reading(list(detectors) + list(motors))
            detector_value = reading[self.calculate_stats_from_detector_name]["value"]
            self._stats_buffer.append(detector_value)
            self._stop_plan = self._stop_plan_criteria(self._stats_buffer)

    def _stop_plan_criteria(self, stats_list: list) -> bool:
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

    def _plot_results(self) -> None:
        """
        Plots the fitted curve along with the raw data and the fit parameters

        Returns
        -------
        None
        """
        if self.calculate_first_derivative:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[2 * 4 * golden_ratio, 4])
            ax[0].scatter(
                self.updated_motor_positions, self.intensity_array, label="Raw data"
            )
            ax[0].set_xlabel(self.motor.name)
            ax[0].set_ylabel(self.calculate_stats_from_detector_name)
            ax[0].legend()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[4 * golden_ratio, 4])
            ax = [None, ax]

        x_tmp = np.linspace(
            min(self.updated_motor_positions), max(self.updated_motor_positions), 4096
        )
        y_tmp = (
            self.statistics.skewnorm_fit_parameters.pdf_scaling_constant
            * skewnorm.pdf(
                x_tmp,
                self.statistics.skewnorm_fit_parameters.a,
                loc=self.statistics.skewnorm_fit_parameters.location,
                scale=self.statistics.skewnorm_fit_parameters.scale,
            )
        )
        if self.calculate_first_derivative:
            ax[1].scatter(
                self.updated_motor_positions,
                self.first_derivative,
                label=r"$\frac{df}{dx}$",
            )
            ax[1].set_xlabel(self.motor.name)
            ax[1].set_ylabel(r"$\frac{df}{dx}$")
        else:
            ax[1].scatter(
                self.updated_motor_positions,
                self.intensity_array,
                label=r"Raw Data",
            )
            ax[1].set_xlabel(self.motor.name)
            ax[1].set_ylabel(self.calculate_stats_from_detector_name)

        mean = round(self.statistics.mean, 2)
        peak = (round(self.statistics.peak[0], 2), round(self.statistics.peak[1], 2))
        sigma = round(self.statistics.sigma, 2)
        skewness = round(self.statistics.skewness, 2)
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
            self.statistics.mean, linestyle="--", label=f"$\mu={mean}$", color="gray"
        )
        ax[1].scatter(
            self.statistics.peak[0],
            self.statistics.peak[1],
            label=f"Peak={peak}",
            marker="2",
            s=300,
            color="k",
        )
        if self.statistics.FWHM is not None:
            FWHM = round(self.statistics.FWHM, 2)
            plt.axvspan(
                xmin=self.statistics.FWHM_x_coords[0],
                xmax=self.statistics.FWHM_x_coords[1],
                alpha=0.2,
                label=f"\nFWHM={FWHM}",
            )
        ax[1].legend()
        plt.tight_layout()
        plt.savefig("stats")
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
        calculate_stats_from_detector_name: str = None,
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
        calculate_stats_from_detector_name : str, optional
            The name of the detector for which statistics should be calculated.
            If not specified, statistics will be calculated for the first detector
            in the detector list by default.
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
        self._filewriter_mode = False

        self.dwell_time = dwell_time
        self.motor_1.settle_time = dwell_time
        self.motor_2.settle_time = dwell_time

        if calculate_stats_from_detector_name is None:
            self.calculate_stats_from_detector_name = detectors[0].name
        else:
            self.calculate_stats_from_detector_name = calculate_stats_from_detector_name

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

        if self.hdf5_filename is None:
            now = datetime.now(tz=timezone.utc)
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            self.hdf5_filename = "mx3_1d_scan_" + dt_string + ".h5"

        if self.metadata is None:
            self.metadata = {"hdf5_filename": self.hdf5_filename}
        elif type(self.metadata) is dict:
            self.metadata.update({"hdf5_filename": self.hdf5_filename})
        elif type(self.metadata) is not dict:
            raise ValueError("Metadata must be a dictionary")

        for index, detector in enumerate(self.detectors):
            detector.kind = "hinted"

            if type(detector) == HDF5Filewriter:
                self._filewriter_mode = True
                detector: HDF5Filewriter
                filewriter_signal_index = index
                yield from mv(
                    detector.filename,
                    self.hdf5_filename,
                    detector.frames_per_datafile,
                    self.number_of_steps_motor_1 * self.number_of_steps_motor_2,
                )
                write_path_template = detector.write_path_template.get()
                self.metadata.update({"write_path_template": write_path_template})

        if not self._filewriter_mode:
            logger.warning(
                "A HDF5Filewriter signal has not been specified in the detector list. "
                "HDF5 files will not be created"
            )

        self.metadata.update({"favourite": False, "favourite_description": ""})
        self._stats_buffer = []
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

        self._plot_heatmap()

        if self._filewriter_mode:
            self._add_metadata_to_hdf5_file(self.detectors[filewriter_signal_index])

    def _add_metadata_to_hdf5_file(
        self, hdf5_filewriter_signal: HDF5Filewriter
    ) -> None:
        """Adds metadata to hdf5 files

        Parameters
        ----------
        hdf5_filewriter_signal : HDF5Filewriter
            A HDF5Filewriter object

        Returns
        -------
        None
        """
        detector_str = []
        for det in self.detectors:
            detector_str.append(det.__str__())

        with h5py.File(hdf5_filewriter_signal.hdf5_path, mode="r+") as f:
            intensity_dataset = f.create_dataset(
                "/entry/data/intensity_array",
                data=np.array(self._stats_buffer).reshape(
                    self.number_of_steps_motor_1, self.number_of_steps_motor_2
                ),
            )
            intensity_dataset.attrs["metadata"] = (
                "x axis corresponds to motor_1 and " + "y axis corresponds to motor_2"
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
        detector_value = reading[self.calculate_stats_from_detector_name]["value"]
        self._stats_buffer.append(detector_value)

    def _plot_heatmap(self) -> None:
        """
        Plots a heatmap

        Returns
        -------
        None
        """
        plt.figure(figsize=[4 * golden_ratio, 4])

        plt.imshow(
            np.array(self._stats_buffer).reshape(
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
        plt.colorbar(label=self.calculate_stats_from_detector_name)
        plt.savefig("stats")
        plt.show()
