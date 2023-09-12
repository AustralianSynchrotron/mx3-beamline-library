import logging
from datetime import datetime, timezone
from typing import Callable, Generator, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluesky.plan_stubs import move_per_step, mv, read, trigger_and_read
from bluesky.plans import scan
from bluesky.utils import Msg
from matplotlib import rc
from ophyd import Signal
from ophyd.areadetector.base import EpicsSignalWithRBV
from ophyd.epics_motor import EpicsMotor
from scipy.constants import golden_ratio
from scipy.stats import skewnorm

from ...devices.classes.detectors import GrasshopperCamera, HDF5Filewriter
from .stats import calculate_1D_scan_stats

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)

rc("xtick", labelsize=13)
rc("ytick", labelsize=13)


class Scan1D:
    """
    This class is used to run a scan on a 1D scan. The resulting distribution is fitted
    using the scipy.stats.skewnorm model
    """

    def __init__(
        self,
        detectors: list[Union[GrasshopperCamera, HDF5Filewriter, EpicsSignalWithRBV]],
        motor: EpicsMotor,
        initial_position: float,
        final_position: float,
        number_of_steps: int,
        metadata: dict = None,
        hdf5_filename: str = None,
        dwell_time: float = 0,
    ) -> None:
        """
        Parameters
        ----------
        detectors : list[Union[GrasshopperCamera, HDF5Filewriter, EpicsSignalWithRBV]]
            A list of detectors, e.g., [my_camera.stats.total, my_camera.stats.sigma].
        motor : EpicsMotor
            The motor being scanned.
        initial_position : float
            The initial position of the motor.
        final_position : float
            The final position of the motor.
        number_of_steps : int
            The number of steps in the scan.
        metadata : dict, optional
            Additional metadata associated with the scan, by default None
        hdf5_filename : str, optional
            Name of the generated HDF5 file. If not provided, the filename is based
            on the generation time.
            Example: "mx3_1d_scan_28-08-2023_05:44:15.h5"
        dwell_time: float, optional
            Amount of time to wait after moves to report status completion, by default 0
        """

        self.motor = motor
        self.detectors = detectors
        self.initial_position = initial_position
        self.final_position = final_position
        self.number_of_steps = number_of_steps
        self.metadata = metadata
        self.hdf5_filename = hdf5_filename

        self.motor.settle_time = dwell_time

        self.motor_array = np.linspace(
            initial_position, final_position, number_of_steps
        )
        self.intensity_array = None

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
            now = datetime.now(tz=timezone.utc)
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            self.hdf5_filename = "mx3_1d_scan_" + dt_string + ".h5"

        if self.metadata is None:
            self.metadata = {"hdf5_filename": self.hdf5_filename}
        elif type(self.metadata) is dict:
            self.metadata.update({"hdf5_filename": self.hdf5_filename})
        elif type(self.metadata) is not dict:
            raise ValueError("Metadata must be a dictionary")

        for detector in self.detectors:
            detector.kind = "hinted"

            if type(detector) == HDF5Filewriter:
                detector: HDF5Filewriter
                yield from mv(
                    detector.filename,
                    self.hdf5_filename,
                    detector.frames_per_datafile,
                    self.number_of_steps,
                )
                write_path_template = detector.write_path_template.get()
                self.metadata.update({"write_path_template": write_path_template})

        # The following signals are used internally to save the total counts as
        # a function of time, and to determine if the plan should be stopped after
        # certain criteria has been met
        _stats_buffer = Signal(name="buffer", kind="omitted", value=[])
        _stop_plan_signal = Signal(name="stop_plan", kind="omitted", value=False)
        self.detectors.append(_stop_plan_signal)
        self.detectors.append(
            _stats_buffer,
        )
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
        self.intensity_array = np.array(_stats_buffer.get())
        self.updated_motor_positions = self.motor_array[: len(self.intensity_array)]

        if len(self.intensity_array) > 4:
            self.statistics = calculate_1D_scan_stats(
                self.updated_motor_positions, self.intensity_array
            )
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
        self._plot_results()

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
        _stop_plan_signal: Signal = detectors[-2]
        if not _stop_plan_signal.get():
            motors = step.keys()
            yield from move_per_step(step, pos_cache)
            reading = yield from take_reading(list(detectors) + list(motors))

            _stats_buffer = detectors[-1]
            for key in reading.keys():
                # TODO: for now we focus on stats.total, but in principle this parameter
                # can be anything we want
                key_index = key.find("total")
                if key_index >= 0:
                    value = reading[key]["value"]
                    stats_list = yield from read(_stats_buffer)
                    stats_list["buffer"]["value"].append(value)
                    yield from mv(_stats_buffer, stats_list["buffer"]["value"])

            stop_plan = self._stop_plan(stats_list["buffer"]["value"])
            yield from mv(_stop_plan_signal, stop_plan)

    def _stop_plan(self, stats_list: list) -> bool:
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
            if len(stats_list) == argmax * 3:
                return True
            else:
                return False

    def _plot_results(self):
        plt.figure(figsize=[5 * golden_ratio, 5])
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
        plt.plot(
            self.updated_motor_positions,
            self.intensity_array,
            label=r"$\bf{" + "Original" + "}$" + r" $\bf{" + "Data" + "}$",
        )
        mean = round(self.statistics.mean, 2)
        peak = round(self.statistics.maximum_y_value, 2)
        if self.statistics.FWHM is not None:
            FWHM = round(self.statistics.FWHM, 2)
            plt.axvspan(
                xmin=self.statistics.FWHM_x_coords[0],
                xmax=self.statistics.FWHM_x_coords[1],
                alpha=0.2,
            )
        else:
            FWHM = None

        sigma = round(self.statistics.sigma, 2)
        skewness = round(self.statistics.skewness, 2)
        label = (
            r"$\bf{"
            + "Curve"
            + "}$"
            + r" $\bf{"
            + "Fit:"
            + "}$"
            + f"\n$\mu={mean}$ \n$\sigma={sigma}$ \npeak={peak} \nskewness={skewness} "
            + f"\nFWHM={FWHM}"
        )
        plt.plot(x_tmp, y_tmp, label=label, linestyle="--")
        plt.legend(fontsize=12)
        plt.xlabel("Motor positions", fontsize=13)
        plt.ylabel("Intensity", fontsize=13)
        plt.show()
