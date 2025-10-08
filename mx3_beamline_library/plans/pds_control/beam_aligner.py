import os
import time
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import Gaussian2dModel
from ophyd import EpicsMotor

from mx3_beamline_library.devices.beam import (
    filter_wheel_is_moving,
    transmission,
    x_RBV,
    y_RBV,
)
from mx3_beamline_library.devices.classes.motors import ASBrickMotor
from mx3_beamline_library.devices.motors import md3
from mx3_beamline_library.logger import setup_logger
from mx3_beamline_library.plans.image_analysis import get_image_from_md3_camera

logger = setup_logger()

# scale for camera in pixels/mm
SCALE_FACTOR = 8502.362


class BeamAligner:
    def __init__(
        self,
        goniometer_x: ASBrickMotor | EpicsMotor,
        goniometer_y: ASBrickMotor | EpicsMotor,
        align_beam_filename: None | str = None,
        md3_snapshot_filename: None | str = None,
        filepath: str = "/mnt/disk/commissioning/notebooks/scandata/beamAlignmentData/",
    ):
        """
        Parameters
        ----------
        goniometer_x : ASBrickMotor | EpicsMotor
            Goniometer x motor obtained from CSBS
        goniometer_y : ASBrickMotor | EpicsMotor
            Goniometer y motor obtained from CSBS
        align_beam_filename : str, optional
            Filename to save alignment data to, by default None. If None, a timestamped filename
            will be generated. This file will be saved in the directory specified by `filepath`.
        md3_snapshot_filename : str, optional
            Filename to save MD3 snapshot to, by default None. If None, a timestamped filename
            will be generated. This file will be saved in the directory specified by `filepath`.
        filepath : str, optional
            Path to save alignment data file, by default
            "/mnt/disk/commissioning/notebooks/scandata/beamAlignmentData/"
        """
        self.scale_factor = SCALE_FACTOR
        self.inposx = 0
        self.inposy = 0
        md3.zoom.set(7)
        self.goni_x = goniometer_x
        self.goni_y = goniometer_y
        self.filepath = filepath
        self.align_beam_filename = align_beam_filename
        self.md3_snapshot_filename = md3_snapshot_filename

    def get_md3_image(self):
        self.im = get_image_from_md3_camera(np.uint16)

        self.crosshair_x = self.im.shape[1] / 2
        self.crosshair_y = self.im.shape[0] / 2

    def make_lmfit_data(self):
        x1 = np.arange(self.im.shape[0])
        x2 = np.arange(self.im.shape[1])
        self.x1n = []
        self.x2n = []
        for i in x1:
            for j in x2:
                self.x1n.append(i)
                self.x2n.append(j)
        self.x1n = np.asarray(self.x1n)
        self.x2n = np.asarray(self.x2n)
        self.data = self.im.flatten()
        self.error = np.sqrt(self.data + 1)

    def new_make_lmfit_data(self):
        X, Y = np.meshgrid(
            np.arange(0, self.im.shape[0], 1), np.arange(0, self.im.shape[1], 1)
        )
        self.x1n = X.flatten()
        self.x2n = Y.flatten()
        self.data = self.im.flatten()
        self.error = np.sqrt(self.data + 1)

    def run_lmfit(self):
        # model = lmfit.models.Gaussian2dModel()
        # params = model.guess(self.data, self.x2n, self.x1n)
        # self.result = model.fit(self.data, x=self.x2n,
        # y=self.x1n, params=params, weights=1/self.error)

        ny, nx = self.im.shape
        x, y = np.arange(nx), np.arange(ny)
        xmesh, ymesh = np.meshgrid(x, y)

        model = Gaussian2dModel(independent_vars=["x", "y"])
        params = model.guess(self.im.flatten(), xmesh.flatten(), ymesh.flatten())
        self.result = model.fit(
            self.im.flatten(), params, x=xmesh.flatten(), y=ymesh.flatten()
        )

        self.imgfit = self.result.best_fit.reshape((ny, nx))

        self.centerx = self.result.best_values["centerx"]
        self.centery = self.result.best_values["centery"]
        logger.info(f"CenterX is {self.centerx}, CenterY is {self.centery}")
        # report=print(self.result.fit_report())

    def calc_moves(self):
        self.deltax = self.crosshair_x - self.centerx
        self.deltay = self.crosshair_y - self.centery
        logger.info(f"deltax is {self.deltax} pixels, deltay is {self.deltay} pixels")
        self.shiftx = self.deltax / self.scale_factor
        self.shifty = self.deltay / self.scale_factor
        logger.info(
            f"required horizontal shift is {self.shiftx * 1e3} microns, "
            f"vertical is {self.shifty * 1e3} microns"
        )

    def check_deadband(self):
        if np.abs(self.shiftx) <= 0.00015:
            logger.info("not moving X as less than 0.15 microns")
            self.shiftx = 0
            self.inposx = 1
        if np.abs(self.shifty) <= 0.00015:
            logger.info("not moving Y as less than 0.15 microns")
            self.shifty = 0
            self.inposy = 1

    def scale_move(self):
        if np.abs(self.shiftx) <= 0.001:
            self.shiftx = self.shiftx * 0.5
        if np.abs(self.shifty) <= 0.001:
            self.shifty = self.shifty * 0.5

    def align_goni(self):

        current_x = self.goni_x.position
        time.sleep(0.1)
        if not self.shiftx == 0:
            self.goni_x.set(current_x + self.shiftx)
            self._wait(self.goni_x)
            # RE(mv(self.goni_x, current_x+self.shiftx))
            time.sleep(2)
        current_y = self.goni_y.position
        time.sleep(0.1)
        if not self.shifty == 0:
            self.goni_y.set(current_y + self.shifty)
            self._wait(self.goni_y)
            # RE(mv(self.goni_y, current_y+self.shifty))
            time.sleep(2)

    def run_loop(self):
        self.set_filter_imaging()
        # self.toggle_fast_shutter("open")
        # time.sleep(2)
        # self.insert_scintillator()
        self.get_md3_image()
        self.make_lmfit_data()
        self.run_lmfit()
        # time.sleep(1)
        self.calc_moves()
        self.check_deadband()
        self.scale_move()
        self.align_goni()
        # self.toggle_fast_shutter("close")

    def get_frame_and_calc_deltas(self):
        self.get_bpm4_positions()
        time.sleep(1)
        self.set_filter_imaging()
        self.toggle_fast_shutter("open")
        # time.sleep(1)
        self.get_md3_image()
        # self.make_lmfit_data()
        self.run_lmfit()
        self.calc_moves()
        self.write_to_datafile()
        self.toggle_fast_shutter("close")

    def write_to_datafile(self):
        self.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_result = f"\n{self.now},{self.shiftx * 1e3},{self.shifty * 1e3},{self.bpm4_Xpos},{self.bpm4_Ypos}"  # noqa
        names = "time,shiftx,shifty,bpm4_X,bpm4_Y"
        if self.align_beam_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.align_beam_filename = f"align_beam_{timestamp}.csv"

        self.fine_fileout = os.path.join(self.filepath, self.align_beam_filename)
        logger.info(f"Writing alignment data to {self.fine_fileout}")
        if not os.path.exists(self.fine_fileout):
            with open(self.fine_fileout, "a") as scan_out:
                scan_out.write(names)
        with open(self.fine_fileout, "a") as scan_out:
            scan_out.write(scan_result)

    def run_alignment_loop(self):
        self.aligned = 0
        self.toggle_fast_shutter("open")
        for _ in range(10):
            self.run_loop()
            time.sleep(1)
            if self.inposx == 1 and self.inposy == 1:
                logger.info("Stopping as beam in position:")
                # self.toggle_fast_shutter("close")
                self.aligned = 1
                break
        if self.aligned != 1:
            logger.warning("did not converge in 10 cycles; exiting")
        # self.toggle_fast_shutter("close")

    def set_max_zoom(self):
        md3.zoom.set(7)

    def toggle_fast_shutter(self, mode: Literal["open", "close"]):
        if mode == "close":
            md3.fast_shutter.set(0)
        elif mode == "open":
            md3.fast_shutter.set(1)
            time.sleep(2)
        else:
            raise KeyError

    def snap_md3_frame(self):
        output_folder = self.filepath
        if self.md3_snapshot_filename is None:
            time_now = time.strftime("%Y%m%d-%H%M%S")
            self.md3_snapshot_filename = os.path.join(time_now + "_md3_beam.png")

        im = get_image_from_md3_camera(np.uint16)
        crosshair_x = 612
        crosshair_y = 512
        crosshair_size = 40
        im = get_image_from_md3_camera(np.uint16)
        plt.imshow(im)
        plt.hlines(
            crosshair_y,
            (crosshair_x - crosshair_size),
            (crosshair_x + crosshair_size),
            colors="red",
            linewidth=1,
        )
        plt.vlines(
            crosshair_x,
            (crosshair_y - crosshair_size),
            (crosshair_y + crosshair_size),
            colors="red",
            linewidth=1,
        )
        output_path = os.path.join(output_folder, self.md3_snapshot_filename)
        logger.info(f"Saving MD3 snapshot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    def retract_scintillator(self):
        md3.scintillator_vertical.set(-2)

    def insert_scintillator(self):
        md3.scintillator_vertical.set(-0.2199)

    def set_filter_steering(self):
        transmission.set(0.05)
        time.sleep(1)
        self.wait_for_filter_wheel()

    def set_filter_imaging(self):
        transmission.set(0.000125)
        time.sleep(1)
        self.wait_for_filter_wheel()

    def get_bpm4_positions(self):
        # TODO: double check, x_RBV and y_RBV correspond to bpm4
        self.bpm4_Xpos = x_RBV.get()
        self.bpm4_Ypos = y_RBV.get()

    def wait_for_filter_wheel(self):
        while True:
            if filter_wheel_is_moving.get() == 0:
                break
            time.sleep(0.1)

    def cycle_steering_and_measure(self):
        # self.set_filter_imaging()
        # self.insert_scintillator()
        # time.sleep(1)
        self.get_frame_and_calc_deltas()
        self.snap_md3_frame()
        # self.retract_scintillator()
        self.set_filter_steering()
        time.sleep(15)

    def get_frame_for_scans(self):
        # self.get_bpm4_positions()
        # time.sleep(1)
        # self.set_filter_imaging()
        # self.toggle_fast_shutter("open")
        # time.sleep(1)
        self.get_md3_image()
        # self.make_lmfit_data()
        self.run_lmfit()
        # self.calc_moves()
        # self.write_to_datafile()
        # self.toggle_fast_shutter("close")

    def _wait(self, motor: ASBrickMotor | EpicsMotor) -> None:
        time.sleep(0.1)
        while motor.moving:
            time.sleep(0.1)


if __name__ == "__main__":
    # NOTE: to run this in simulation mode self.im
    # has to be replaced with
    # self.im = np.load(
    # "/home/fhv/bitbucket_repos/mx-prefect-bluesky-work-pool/2d_gaussian/md3_image_for_FHV.npy" # noqa
    # )
    # That is because this class requires a black and white image, and the simulated
    # camera returns a color image
    from mx3_beamline_library.devices.sim.motors import MX3SimMotor

    goni_x = MX3SimMotor(name="goni_x")
    goni_y = MX3SimMotor(name="goni_y")

    beam_aligner = BeamAligner(
        goniometer_x=goni_x,
        goniometer_y=goni_y,
        # align_beam_filename="test_align.csv",
        # md3_snapshot_filename="test_md3.png",
        filepath="./",
    )
    beam_aligner.snap_md3_frame()
    beam_aligner.run_alignment_loop()
    beam_aligner.get_frame_and_calc_deltas()
