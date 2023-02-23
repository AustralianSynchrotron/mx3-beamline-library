import asyncio
import logging
import os
import pickle
from os import environ
from typing import Generator, Optional, Union

import httpx
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import redis
import yaml
from bluesky.plan_stubs import mv, open_run, close_run, monitor
from bluesky.preprocessors import monitor_during_decorator, monitor_during_wrapper, run_wrapper
from bluesky.utils import Msg
from pydantic import ValidationError

from mx3_beamline_library.devices.classes.detectors import (
    BlackFlyCam,
    DectrisDetector,
    MDRedisCam,
)
from mx3_beamline_library.devices.classes.motors import (
    CosylabMotor,
    MD3BackLight,
    MD3Motor,
    MD3Phase,
    MD3Zoom,
)
from mx3_beamline_library.plans.basic_scans import md3_4d_scan, md3_grid_scan, arm_trigger_and_disarm_detector
from mx3_beamline_library.plans.optical_centering import OpticalCentering
from mx3_beamline_library.schemas.optical_and_xray_centering import (
    BlueskyEventDoc,
    RasterGridMotorCoordinates,
    SpotfinderResults,
    TestrigEventData,
)
from mx3_beamline_library.science.optical_and_loop_centering.crystal_finder import (
    CrystalFinder, CrystalFinder3D
)
from mx3_beamline_library.science.optical_and_loop_centering.psi_optical_centering import (
    loopImageProcessing,
)
from ophyd import Signal
from ..schemas.optical_and_xray_centering import MD3ScanResponse

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


class OpticalAndXRayCentering(OpticalCentering):
    """
    This plan consists of different steps explained as follows:
    1) Center the loop using the methods described in the OpticalCentering class.
    2) Prepare a raster grid, execute a raster scan for the surface of the loop where
    its area is largest
    3) Find the crystals using the CrystalFinder.
    4) Rotate the loop 90 degrees and repeat steps 2 and 3, to determine the
    location of the crystals in the orthogonal plane, which allows us to finally
    infer the positions of the crystals in 3D.
    """

    def __init__(
        self,
        detector: DectrisDetector,
        camera: Union[BlackFlyCam, MDRedisCam],
        sample_x: Union[CosylabMotor, MD3Motor],
        sample_y: Union[CosylabMotor, MD3Motor],
        alignment_x: Union[CosylabMotor, MD3Motor],
        alignment_y: Union[CosylabMotor, MD3Motor],
        alignment_z: Union[CosylabMotor, MD3Motor],
        omega: Union[CosylabMotor, MD3Motor],
        zoom: MD3Zoom,
        phase: MD3Phase,
        backlight: MD3BackLight,
        metadata: dict,
        beam_position: tuple[int, int],
        auto_focus: bool = True,
        min_focus: float = -0.3,
        max_focus: float = 1.3,
        tol: float = 0.3,
        number_of_intervals: int = 2,
        plot: bool = False,
        loop_img_processing_beamline: str = "MX3",
        loop_img_processing_zoom: str = "1",
        number_of_omega_steps: int = 5,
        threshold: int = 20,
        beam_size: tuple[float, float] = (100.0, 100.0),
        exposure_time: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        detector: DectrisDetector
            The dectris detector ophyd device
        camera : Union[BlackFlyCam, MDRedisCam]
            Camera
        sample_x : Union[CosylabMotor, MD3Motor]
            Sample x
        sample_y : Union[CosylabMotor, MD3Motor]
            Sample y
        alignment_x : Union[CosylabMotor, MD3Motor]
            Alignment x
        alignment_y : Union[CosylabMotor, MD3Motor]
            Alignment y
        alignment_z : Union[CosylabMotor, MD3Motor]
            Alignment y
        omega : Union[CosylabMotor, MD3Motor]
            Omega
        zoom : MD3Zoom
            Zoom
        phase : MD3Phase
            MD3 phase ophyd-signal
        backlight : MD3Backlight
            Backlight
        metadata : dict
            Bluesky metadata, we include here the sample id,
            e.g. {"sample_id": "test_sample"}
        beam_position : tuple[int, int]
            Position of the beam in units of pixels
        auto_focus : bool, optional
            If true, we autofocus the image before analysing an image,
            by default True
        min_focus : float, optional
            Minimum value to search for the maximum of var( Img * L(x,y) ),
            by default -0.3
        max_focus : float, optional
            Maximum value to search for the maximum of var( Img * L(x,y) ),
            by default 1.3
        tol : float, optional
            The tolerance used by the Golden-section search, by default 0.3
        number_of_intervals : int, optional
            Number of intervals used to find local maximums of the function
            `var( Img * L(x,y) )`, by default 2
        plot : bool, optional
            If true, we take snapshots of the plan at different stages for debugging purposes.
            By default false
        loop_img_processing_beamline : str
            This name is used to get the configuration parameters used by the
            loop image processing code developed by PSI, by default testrig
        loop_img_processing_zoom : str
            We get the configuration parameters used by the loop image processing code
            developed by PSI, for a particular zoom, by default 1.0
        number_of_omega_steps : int, optional
            Number of omega values used to find the edge and flat surface of the loop,
            by default 5
        threshold : float
            This parameter is used by the CrystalFinder class. Below this threshold,
            we replace all numbers of the number_of_spots array obtained from
            the grid scan plan with zeros.
        beam_size : tuple[float, float]
            We assume that the shape of the beam is a rectangle of length (x, y),
            where x and y are the width and height of the rectangle respectively.
            The beam size is measured in units of micrometers
        exposure_time : float
            Detector exposure time

        Returns
        -------
        None
        """
        super().__init__(
            camera,
            sample_x,
            sample_y,
            alignment_x,
            alignment_y,
            alignment_z,
            omega,
            zoom,
            phase,
            backlight,
            beam_position,
            auto_focus,
            min_focus,
            max_focus,
            tol,
            number_of_intervals,
            plot,
            loop_img_processing_beamline,
            loop_img_processing_zoom,
            number_of_omega_steps,
        )
        self.detector = detector
        self.metadata = metadata
        self.threshold = threshold
        self.beam_size = beam_size
        self.exposure_time = exposure_time

        REDIS_HOST = environ.get("REDIS_HOST", "0.0.0.0")
        REDIS_PORT = int(environ.get("REDIS_PORT", "6379"))
        self.redis_connection = redis.StrictRedis(
            host=REDIS_HOST, port=REDIS_PORT, db=0
        )

        self.mxcube_url = environ.get("MXCUBE_URL", "http://localhost:8090")

        self.grid_id = 0

        self.grid_scan_coordinates_flat = Signal(name="grid_scan_coordinates_flat", kind="normal")
        self.grid_scan_coordinates_edge = Signal(name="grid_scan_coordinates_edge", kind="normal")
        self.md3_scan_response = Signal(name="md3_scan_response", kind="normal")

    def start(self) -> Generator[Msg, None, None]:
        """
        This is the plan that we call to run optical and x ray centering.
        It based on the centering code defined in Fig. 2 of
        Hirata et al. (2019). Acta Cryst. D75, 138-150.

        Yields
        ------
        Generator[Msg, None, None]
            A bluesky plan tha centers the a sample using optical and X-ray centering
        """
        # Step 1: Mount the sample (NOT IMPLEMENTED)

        # Step 2: Loop centering
        logger.info("Step 2: Loop centering")
        # yield from open_run(md=self.metadata)
        # yield from self.monitor_signals()
            

        yield from self.center_loop()

        # Step 3: Prepare raster grids for the edge surface
        yield from mv(self.zoom, 4, self.omega, self.edge_angle)
        grid_edge, _ = self.prepare_raster_grid("step_3_prep_raster_grid_edge")
        # Add metadata for bluesky documents
        self.grid_scan_coordinates_edge.put(grid_edge.dict())

        # Step 3: Prepare raster grids for the flat surface
        yield from mv(self.zoom, 4, self.omega, self.flat_angle)
        grid_flat, rectangle_coordinates_flat = self.prepare_raster_grid(
            "step_3_prep_raster_grid_flat"
        )
        # Add metadata for bluesky documents
        self.grid_scan_coordinates_flat.put(grid_flat.dict())

        # Steps 4 to 6:
        logger.info(
            "Steps 4-6: Execute raster scan, and find crystals for the flat surface of the loop"
        )
        yield from mv(self.omega, self.flat_angle)
        (
            centers_of_mass_flat,
            crystal_locations_flat,
            distances_between_crystals_flat,
            last_id,
        ) = yield from self.start_raster_scan_and_find_crystals(
            grid_flat, grid_scan_type="flat", last_id=0, filename="flat"
        )

        # Step 7: Rotate loop 90 degrees, repeat steps 3 to 6
        logger.info("Step 7: Repeat steps 4 to 6 for the edge surface")
        yield from mv(self.omega, self.edge_angle)
        (
            centers_of_mass_edge,
            crystal_locations_edge,
            distances_between_crystals_edge,
            last_id,
        ) = yield from self.start_raster_scan_and_find_crystals(
            grid_edge,
            grid_scan_type="edge",
            last_id=last_id,
            filename="edge",
            draw_grid_in_mxcube=True,
            rectangle_coordinates_in_pixels=rectangle_coordinates_flat,
        )

        # Step 8: Infer location of crystals in 3D
        logger.info("Step 8: Infer location of crystals in 3D")
        crystal_finder_3d = CrystalFinder3D(
            crystal_locations_flat,
            crystal_locations_edge,
            centers_of_mass_flat,
            centers_of_mass_edge,
            distances_between_crystals_flat,
            distances_between_crystals_edge,
        )
        crystal_finder_3d.plot_crystals(plot_centers_of_mass=True, save=self.plot)

        #yield from close_run(
        #    exit_status="success", 
        #    reason="Optical and x ray centering was executed successfully"
        #)


    def start_raster_scan_and_find_crystals(
        self,
        grid: RasterGridMotorCoordinates,
        grid_scan_type: str,
        last_id: Union[int, bytes],
        filename: str = "crystal_finder_results",
        draw_grid_in_mxcube: bool = False,
        rectangle_coordinates_in_pixels: dict = None,
    ) -> tuple[list[tuple[int, int]], list[dict], list[dict[str, int]], bytes]:
        """
        Prepares the raster grid, executes the raster plan, and finds the crystals
        in a loop using the CrystalFinder. This method is reused to analyse the
        flat and edge surfaces of the loop.

        Parameters
        ----------
        grid : RasterGridMotorCoordinates
            A RasterGridMotorCoordinates object which contains information about the
            raster grid, including its width, height and initial and final positions
            of sample_x, sample_y, and alignment_y
        grid_scan_type : str
            The grid scan type. Can be either `flat` or `edge`
        last_id : Union[int, bytes]
            Redis streams last_id
        filename : str, optional
            Name of the file used to save the CrystalFinder results if self.plot=True,
            by default "crystal_finder_results"
        draw_grid_in_mxcube : bool
            If true, we draw a grid in mxcube, by default False
        rectangle_coordinates_in_pixels : dict
            Rectangle coordinates in pixels
        Returns
        -------
        tuple[list[tuple[int, int]], list[dict], list[dict[str, int]], bytes]
            A list containing the centers of mass of all crystals in the loop,
            a list of dictionaries containing information about the locations and sizes
            of all crystals,
            a list of dictionaries describing the distance between all overlapping crystals,
            and the updated redis streams last_id

        """
        logger.info("Starting raster scan...")
        logger.info(f"Number of columns: {grid.number_of_columns}")
        logger.info(f"Number of rows: {grid.number_of_rows}")
        logger.info(f"Grid width [mm]: {grid.width}")
        logger.info(f"Grid height [mm]: {grid.height}")

        # NOTE: we hardcode nimages at the moment to be able to properly 
        # analyse the data with the spotfinder.
        self.metadata.update({"grid_scan_type": grid_scan_type})
        detector_configuration = {
            "nimages": grid.number_of_columns* grid.number_of_rows, 
            "user_data": self.metadata
        } 
        # TODO: Is the detector config determined by the user or set by default
        # for any UDC experiment?

        # NOTE: The md3_grid_scan does not like number_of_columns < 2. If
        # number_of_columns < 2 we use the md3_3d_scan instead, setting scan_range=0,
        # and keeping the values of sample_x, sample_y, and alignment_z constant
        if environ["BL_ACTIVE"].lower() == "true":
            if grid.number_of_columns >= 2:
                scan_response = yield from md3_grid_scan(
                    detector=self.detector,
                    detector_configuration=detector_configuration, # this is not used
                    metadata={"sample_id": "sample_test"},
                    grid_width=grid.width,
                    grid_height=grid.height,
                    number_of_columns=grid.number_of_columns,
                    number_of_rows=grid.number_of_rows,
                    start_omega=self.omega.position,
                    start_alignment_y=grid.initial_pos_alignment_y,
                    start_alignment_z=self.centered_loop_position.alignment_z,
                    start_sample_x=grid.final_pos_sample_x,
                    start_sample_y=grid.final_pos_sample_y,
                    exposure_time=self.exposure_time,
                )
            else:
                scan_response = yield from md3_4d_scan(
                    detector=self.detector,
                    detector_configuration=detector_configuration,
                    metadata={"sample_id": "sample_test"}, # This is not currently used
                    start_angle=self.omega.position,
                    scan_range=0,
                    exposure_time=self.exposure_time,
                    start_alignment_y=grid.initial_pos_alignment_y,
                    stop_alignment_y=grid.final_pos_alignment_y,
                    start_sample_x=grid.center_pos_sample_x,
                    stop_sample_x=grid.center_pos_sample_x,
                    start_sample_y=grid.center_pos_sample_y,
                    stop_sample_y=grid.center_pos_sample_y,
                    start_alignment_z=self.centered_loop_position.alignment_z,
                    stop_alignment_z=self.centered_loop_position.alignment_z,
                )
        elif environ["BL_ACTIVE"].lower() == "false":
            # Trigger the simulated simplon api, and return
            # a random MD3ScanResponse
            yield from arm_trigger_and_disarm_detector(
                detector=self.detector, 
                detector_configuration=detector_configuration, 
                metadata=self.metadata
            )
            scan_response = MD3ScanResponse(
                task_name='Raster Scan', 
                task_flags=8, 
                start_time='2023-02-21 12:40:47.502', 
                end_time='2023-02-21 12:40:52.814', 
                task_output='org.embl.dev.pmac.PmacDiagnosticInfo@64ba4055', 
                task_exception='null', 
                result_id=1
            )

        self.md3_scan_response.put(scan_response.dict())

        # Find crystals
        logger.info("Finding crystals...")
        (
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
            number_of_spots_array,
            last_id,
        ) = self.find_crystal_positions(
            self.metadata["sample_id"],
            grid_scan_type=grid_scan_type,
            last_id=last_id,
            n_rows=grid.number_of_rows,
            n_cols=grid.number_of_columns,
            filename=filename,
        )

        if draw_grid_in_mxcube:
            loop = asyncio.get_event_loop()
            loop.create_task(
                self.draw_grid_in_mxcube(
                    rectangle_coordinates_in_pixels,
                    grid.number_of_columns,
                    grid.number_of_rows,
                )
            )

        return (  # noqa
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
            last_id,
        )

    def prepare_raster_grid(
        self, filename: str = "step_3_prep_raster"
    ) -> tuple[RasterGridMotorCoordinates, dict]:
        """
        Prepares a raster grid. The limits of the grid are obtained using
        the PSI loop centering code

        Parameters
        ----------
        filename: str
            Name of the file used to plot save the results if self.plot = True,
            by default step_3_prep_raster

        Returns
        -------
        motor_coordinates: RasterGridMotorCoordinates
            A pydantic model containing the initial and final motor positions of the grid.
        rectangle_coordinates: dict
            Rectangle coordinates in pixels
        """
        # the loopImageProcessing code only works with np.uint8 data types
        data = self.get_image_from_camera(np.uint8)

        procImg = loopImageProcessing(data)
        procImg.findContour(
            zoom=self.loop_img_processing_zoom,
            beamline=self.loop_img_processing_beamline,
        )
        procImg.findExtremes()
        rectangle_coordinates = procImg.fitRectangle()

        if self.plot:
            self.plot_raster_grid(
                rectangle_coordinates,
                filename,
            )

        # Width and height of the grid in (mm)
        width = (
            abs(
                rectangle_coordinates["top_left"][0]
                - rectangle_coordinates["bottom_right"][0]
            )
            / self.zoom.pixels_per_mm
        )
        height = (
            abs(
                rectangle_coordinates["top_left"][1]
                - rectangle_coordinates["bottom_right"][1]
            )
            / self.zoom.pixels_per_mm
        )

        # Y pixel coordinates
        initial_pos_y_pixels = abs(
            rectangle_coordinates["top_left"][1] - self.beam_position[1]
        )
        final_pos_y_pixels = abs(
            rectangle_coordinates["bottom_right"][1] - self.beam_position[1]
        )

        # Alignment y target positions (mm)
        initial_pos_alignment_y = (
            self.alignment_y.position - initial_pos_y_pixels / self.zoom.pixels_per_mm
        )
        final_pos_alignment_y = (
            self.alignment_y.position + final_pos_y_pixels / self.zoom.pixels_per_mm
        )

        # X pixel coordinates
        initial_pos_x_pixels = abs(
            rectangle_coordinates["top_left"][0] - self.beam_position[0]
        )
        final_pos_x_pixels = abs(
            rectangle_coordinates["bottom_right"][0] - self.beam_position[0]
        )

        # Sample x target positions (mm)
        initial_pos_sample_x = self.sample_x.position - np.sin(
            np.radians(self.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_x = self.sample_x.position + np.sin(
            np.radians(self.omega.position)
        ) * (+final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Sample y target positions (mm)
        initial_pos_sample_y = self.sample_y.position - np.cos(
            np.radians(self.omega.position)
        ) * (initial_pos_x_pixels / self.zoom.pixels_per_mm)
        final_pos_sample_y = self.sample_y.position + np.cos(
            np.radians(self.omega.position)
        ) * (final_pos_x_pixels / self.zoom.pixels_per_mm)

        # Center of the grid (mm) (y-axis only)
        center_x_of_grid_pixels = (
            rectangle_coordinates["top_left"][0]
            + rectangle_coordinates["bottom_right"][0]
        ) / 2
        center_pos_sample_x = self.sample_x.position + np.sin(
            np.radians(self.omega.position)
        ) * (
            (center_x_of_grid_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )
        center_pos_sample_y = self.sample_y.position + np.cos(
            np.radians(self.omega.position)
        ) * (
            (center_x_of_grid_pixels - self.beam_position[0]) / self.zoom.pixels_per_mm
        )

        # NOTE: The width and height are measured in mm and the beam_size in micrometers,
        # hence the conversion below
        number_of_columns = int(width / (self.beam_size[0] / 1000))
        number_of_rows = int(height / (self.beam_size[1] / 1000))

        motor_coordinates = RasterGridMotorCoordinates(
            initial_pos_sample_x=initial_pos_sample_x,
            final_pos_sample_x=final_pos_sample_x,
            initial_pos_sample_y=initial_pos_sample_y,
            final_pos_sample_y=final_pos_sample_y,
            initial_pos_alignment_y=initial_pos_alignment_y,
            final_pos_alignment_y=final_pos_alignment_y,
            width=width,
            height=height,
            center_pos_sample_x=center_pos_sample_x,
            center_pos_sample_y=center_pos_sample_y,
            number_of_columns=number_of_columns,
            number_of_rows=number_of_rows
        )
        logger.info(f"Raster grid coordinates [mm]: {motor_coordinates}")

        return motor_coordinates, rectangle_coordinates

    def find_crystal_positions(
        self,
        sample_id: str,
        grid_scan_type: str,
        last_id: Union[int, bytes],
        n_rows: int,
        n_cols: int,
        filename: str = "crystal_finder_results",
    ) -> tuple[
        list[tuple[int, int]], list[dict], list[dict[str, int]], npt.NDArray, bytes
    ]:
        """
        Finds the crystal position based on the number of spots obtained from a
        grid_scan using the CrystalFinder class. The number of spots are obtained
        from redis streams, which are generated by the mx - spotfinder in the
        mx - zmq - consumer.

        Parameters
        ----------
        sample_id: str
            Sample id
        grid_scan_type : str
            The grid scan type. Can be either `flat` or `edge`
        last_id: Union[int, bytes]
            Redis streams last_id
        n_rows: int
            Number of rows of the grid
        n_cols: int
            Number of columns of the grid
        filename: str, optional
            The name of the file used to save the CrystalFinder results if
            self.plot = True, by default crystal_finder_results

        Returns
        -------
        tuple[list[tuple[int, int]], list[dict],
            list[dict[str, int]], npt.NDArray, bytes]
            A list containing the centers of mass of all crystals in the loop,
            a list of dictionaries containing information about the locations and sizes
            of all crystals,
            a list of dictionaries describing the distance between all overlapping crystals,
            a numpy array containing the numbers of spots, which shape is (n_rows, n_cols),
            and the last id of the redis streams sequence
        """
        result = []
        number_of_spots_list = []
        import time

        number_of_frames = self.redis_connection.xlen(
            f"spotfinder_results_{grid_scan_type}:{sample_id}"
            )
        while number_of_frames != n_rows*n_cols:
            # TODO: include a timeout, and notify the user that we lost frames
            # somewhere
            time.sleep(0.2)
            number_of_frames = self.redis_connection.xlen(
                f"spotfinder_results_{grid_scan_type}:{sample_id}"
            )
            logger.info(f"Expecting {n_rows*n_cols} frames, got {number_of_frames} frames so far")

        for _ in range(number_of_frames):
            try:
                spotfinder_results, last_id = self.get_spotfinder_results(
                    sample_id, grid_scan_type, last_id
                )
                result.append(spotfinder_results)
                number_of_spots_list.append(spotfinder_results.number_of_spots)
            except IndexError:
                pass


        spots_array = np.array(number_of_spots_list).reshape(n_rows, n_cols)
        spotfinder_results_array = np.array(result).reshape(n_rows, n_cols)


        crystal_finder = CrystalFinder(spots_array, threshold=self.threshold)
        (
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
        ) = crystal_finder.plot_crystal_finder_results(
            save=self.plot, filename=filename
        )

        for cm in centers_of_mass:
            logger.info(
                "Spotfinder results evaluated at the centers of mass: "
                f"{spotfinder_results_array[cm[1]][cm[0]]}"
            )

        # TODO: the spotfinder_results_array is useful for relating pixels to
        # actual motor positions since it contains all the parameters described in the
        # SpotfinderResults pydantic model. At this stage, we're just interested in
        # finding the position of the crystal in units of pixels, so we
        # leave translation from pixels to motor positions for the next sprint

        return (
            centers_of_mass,
            crystal_locations,
            distances_between_crystals,
            spots_array,
            last_id,
        )

    def get_spotfinder_results(
        self, sample_id: str, grid_scan_type: str, id: Union[bytes, int]
    ) -> tuple[SpotfinderResults, bytes]:
        """
        Gets the spotfinder results from redis streams. The spotfinder results
        are calculated by the mx-hdf5-builder service. The name of the redis key
        follows the format f"spotfinder_results:{sample_id}"

        Parameters
        ----------
        sample_id : str
            The sample_id, e.g. my_sample
        grid_scan_type : str
            The grid scan type. Can be either `flat` or `edge`
        id: Union[bytes, int]
            id of the topic in bytes or int format

        Returns
        -------
        spotfinder_results, last_id: tuple[SpotfinderResults, bytes]
            A tuple containing SpotfinderResults and the redis streams
            last_id
        """
        topic = f"spotfinder_results_{grid_scan_type}:{sample_id}"
        response = self.redis_connection.xread({topic: id}, count=1)

        # Extract key and messages from the response
        _, messages = response[0]

        # Update last_id and store messages data
        last_id, data = messages[0]

        bluesky_event_doc = pickle.loads(data[b"bluesky_event_doc"])

        spotfinder_results = SpotfinderResults(
            type=data[b"type"],
            number_of_spots=data[b"number_of_spots"],
            image_id=data[b"image_id"],
            sample_id=data[b"sample_id"],
            bluesky_event_doc=bluesky_event_doc,
            grid_scan_type=data[b"grid_scan_type"]
        )

        assert sample_id == spotfinder_results.sample_id, (
            "The spotfinder sample_id is different from the queueserver sample_id"
        )
        return spotfinder_results, last_id

    async def draw_grid_in_mxcube(
        self,
        rectangle_coordinates: dict,
        num_cols: int,
        num_rows: int,
        number_of_spots_array: Optional[npt.NDArray] = None,
    ):
        """Draws a grid in mxcube

        Parameters
        ----------
        rectangle_coordinates: dict
            Rectangle coordinates of the grid obtained from the PSI loop centering code
        num_cols: int
            Number of columns of the grid
        num_rows: int
            Number of rows of the grid
        number_of_spots_array: npt.NDArray, optional
            A numpy array of shape(n_rows, n_cols) containing the
            number of spots of the grid, by default None

        Returns
        -------
        None
        """
        self.grid_id += 1

        if number_of_spots_array is not None:
            heatmap_and_crystalmap = self.create_heatmap_and_crystal_map(
                num_cols, num_rows, number_of_spots_array
            )
        else:
            heatmap_and_crystalmap = []

        width = int(
            rectangle_coordinates["bottom_right"][0]
            - rectangle_coordinates["top_left"][0]
        )  # pixels
        height = int(
            rectangle_coordinates["bottom_right"][1]
            - rectangle_coordinates["top_left"][1]
        )  # pixels

        mm_per_pixel = 1 / self.zoom.pixels_per_mm
        cell_width = (width / num_cols) * mm_per_pixel * 1000  # micrometers
        cell_height = (height / num_rows) * mm_per_pixel * 1000  # micrometers

        # TODO: Maybe it'd be useful to pass accurate info about the motor positions
        # even though it is technically not used to draw the grid
        mxcube_payload = {
            "shapes": [
                {
                    "cellCountFun": "zig-zag",
                    "cellHSpace": 0,
                    "cellHeight": cell_height,
                    "cellVSpace": 0,
                    "cellWidth": cell_width,
                    "height": height,
                    "width": width,
                    "hideThreshold": 5,
                    "id": f"G{self.grid_id}",
                    "label": "Grid",
                    "motorPositions": {
                        "beamX": 0.141828,
                        "beamY": 0.105672,
                        "kappa": 11,
                        "kappaPhi": 22,
                        "phi": 311.1,
                        "phiy": 34.30887849323582,
                        "phiz": 1.1,
                        "sampx": -0.0032739045158179936,
                        "sampy": -1.0605072324693783,
                    },
                    "name": f"Grid-{self.grid_id}",
                    "numCols": num_cols,
                    "numRows": num_rows,
                    "result": heatmap_and_crystalmap,
                    "screenCoord": rectangle_coordinates["top_left"].tolist(),
                    "selected": True,
                    "state": "SAVED",
                    "t": "G",
                    "pixelsPerMm": [self.zoom.pixels_per_mm, self.zoom.pixels_per_mm],
                    # 'dxMm': 1/292.8705182537115,
                    # 'dyMm': 1/292.8705182537115
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    os.path.join(
                        self.mxcube_url,
                        "mxcube/api/v0.1/sampleview/shapes/create_grid",
                    ),
                    json=mxcube_payload,
                )
                logger.info(f"MXCuBE request response: {response.text}")

        except httpx.ConnectError:
            logger.info("MXCuBE is not available, cannot draw grid in MXCuBE")

    def create_heatmap_and_crystal_map(
        self, num_cols: int, num_rows: int, number_of_spots_array: npt.NDArray
    ) -> dict:
        """
        Creates a heatmap from the number of spots, number of columns
        and number of rows of a grid. The crystal map currently returns
        random numbers since at this stage we only calculate the heatmap.

        Parameters
        ----------
        num_cols: int
            Number of columns
        num_rows: int
            Number of rows
        number_of_spots_array: npt.NDArray
            A numpy array of shape(n_rows, n_cols) containing the
            number of spots of the grid

        Returns
        -------
        dict
            A dictionary containing a heatmap and crystal map in rbga format.
            Currently the crystal map is a random array and only the heatmap
            contains meaningful results
        """

        x = np.arange(num_cols)
        y = np.arange(num_rows)

        y, x = np.meshgrid(x, y)

        z_min = np.min(number_of_spots_array)
        z_max = np.max(number_of_spots_array)

        _, ax = plt.subplots()

        heatmap = ax.pcolormesh(
            x, y, number_of_spots_array, cmap="seismic", vmin=z_min, vmax=z_max
        )
        heatmap = heatmap.to_rgba(number_of_spots_array, norm=True).reshape(
            num_cols * num_rows, 4
        )

        # The following could probably be done more efficiently without using for loops
        heatmap_array = np.ones(heatmap.shape)
        for i in range(num_rows * num_cols):
            for j in range(4):
                if heatmap[i][j] != 1.0:
                    heatmap_array[i][j] = int(heatmap[i][j] * 255)

        heatmap_array = heatmap_array.tolist()

        heatmap = {}
        crystalmap = {}

        for i in range(1, num_rows * num_cols + 1):
            heatmap[i] = [i, list(heatmap_array[i - 1])]

            crystalmap[i] = [
                i,
                [
                    int(np.random.random() * 255),
                    int(np.random.random() * 255),
                    int(np.random.random() * 255),
                    1,
                ],
            ]

        return {"heatmap": heatmap, "crystalmap": crystalmap}

    def plot_raster_grid(
        self,
        rectangle_coordinates: dict,
        filename: str,
    ) -> None:
        """
        Plots the limits of the raster grid on top of the image taken from the
        camera.

        Parameters
        ----------
        initial_pos_pixels: list[int, int]
            The x and z coordinates of the initial position of the grid
        final_pos_pixels: list[int, int]
            The x and z coordinates of the final position of the grid
        filename: str
            The name of the PNG file

        Returns
        -------
        None
        """
        plt.figure()
        data = self.get_image_from_camera()
        plt.imshow(data)

        # Plot grid:
        # Top
        plt.scatter(
            rectangle_coordinates["top_left"][0],
            rectangle_coordinates["top_left"][1],
            s=200,
            c="b",
            marker="+",
        )
        plt.scatter(
            rectangle_coordinates["bottom_right"][0],
            rectangle_coordinates["bottom_right"][1],
            s=200,
            c="b",
            marker="+",
        )

        # top
        x = np.linspace(
            rectangle_coordinates["top_left"][0],
            rectangle_coordinates["bottom_right"][0],
            100,
        )
        z = rectangle_coordinates["top_left"][1] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Bottom
        x = np.linspace(
            rectangle_coordinates["top_left"][0],
            rectangle_coordinates["bottom_right"][0],
            100,
        )
        z = rectangle_coordinates["bottom_right"][1] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Right side
        z = np.linspace(
            rectangle_coordinates["top_left"][1],
            rectangle_coordinates["bottom_right"][1],
            100,
        )
        x = rectangle_coordinates["bottom_right"][0] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")

        # Left side
        z = np.linspace(
            rectangle_coordinates["top_left"][1],
            rectangle_coordinates["bottom_right"][1],
            100,
        )
        x = rectangle_coordinates["top_left"][0] * np.ones(len(x))
        plt.plot(x, z, color="red", linestyle="--")
        plt.title(f"Omega = {round(self.omega.position, 2)} [degrees]")
        plt.legend()

        plt.savefig(filename)
        plt.close()


path_to_config_file = os.path.join(
    os.path.dirname(__file__), "configuration/optical_and_xray_centering.yml"
)

with open(path_to_config_file, "r") as plan_config:
    plan_args: dict = yaml.safe_load(plan_config)

def optical_and_xray_centering(
    detector: DectrisDetector,
    camera: Union[BlackFlyCam, MDRedisCam],
    sample_x: Union[CosylabMotor, MD3Motor],
    sample_y: Union[CosylabMotor, MD3Motor],
    alignment_x: Union[CosylabMotor, MD3Motor],
    alignment_y: Union[CosylabMotor, MD3Motor],
    alignment_z: Union[CosylabMotor, MD3Motor],
    metadata: dict,
    omega: Union[CosylabMotor, MD3Motor],
    zoom: MD3Zoom,
    phase: MD3Phase,
    backlight: MD3BackLight,
    beam_position: tuple[int, int],
    beam_size: tuple[float, float] = (100.0, 100),
    exposure_time: float = 1.0,
) -> Generator[Msg, None, None]:
    """
    This is a wrapper to execute the optical and xray centering plan
    using the OpticalAndXRayCentering class. This function is needed because the
    bluesky - queueserver does not interact nicely with classes.
    The default parameters used in the plan are loaded from the
    optical_and_xray_centering.yml file located in the mx3 - beamline - library
    configuration folder.

    Parameters
    ----------
    detector: DectrisDetector
        The dectris detector ophyd device
    camera : Union[BlackFlyCam, MDRedisCam]
        Camera
    sample_x : Union[CosylabMotor, MD3Motor]
        Sample x
    sample_y : Union[CosylabMotor, MD3Motor]
        Sample y
    alignment_x : Union[CosylabMotor, MD3Motor]
        Alignment x
    alignment_y : Union[CosylabMotor, MD3Motor]
        Alignment y
    alignment_z : Union[CosylabMotor, MD3Motor]
        Alignment y
    omega : Union[CosylabMotor, MD3Motor]
        Omega
    zoom : MD3Zoom
        Zoom
    phase : MD3Phase
        MD3 phase ophyd-signal
    backlight : MD3Backlight
        Backlight
    metadata : dict
        Bluesky metadata, we include here the sample id,
        e.g. {"sample_id": "test_sample"}
    beam_position : tuple[int, int]
        Position of the beam in units of pixels
    beam_size : tuple[float, float]
        We assume that the shape of the beam is a rectangle of length (x, y),
        where x and y are the width and height of the rectangle respectively.
        The beam size is measured in units of micrometers
    exposure_time : float
        Detector exposure time

    Returns
    -------
    Generator[Msg, None, None]
    """
    logger.info(
        f"Plan default arguments obtained from the yaml configuration file: {plan_args}"
    )

    threshold: float = plan_args["crystal_finder"]["threshold"]
    loop_img_processing_beamline: str = plan_args["loop_image_processing"]["beamline"]
    loop_img_processing_zoom: str = plan_args["loop_image_processing"]["zoom"]
    auto_focus: bool = plan_args["autofocus_image"]["autofocus"]
    min_focus: float = plan_args["autofocus_image"]["min"]
    max_focus: float = plan_args["autofocus_image"]["max"]
    tol: float = plan_args["autofocus_image"]["tol"]
    plot: bool = plan_args["plot_results"]
    number_of_intervals: float = plan_args["autofocus_image"]["number_of_intervals"]
    number_of_omega_steps: float = plan_args["loop_area_estimation"][
        "number_of_omega_steps"
    ]

    _optical_and_xray_centering = OpticalAndXRayCentering(
        detector=detector,
        camera=camera,
        sample_x=sample_x,
        sample_y=sample_y,
        alignment_x=alignment_x,
        alignment_y=alignment_y,
        alignment_z=alignment_z,
        omega=omega,
        zoom=zoom,
        phase=phase,
        backlight=backlight,
        beam_position=beam_position,
        auto_focus=auto_focus,
        min_focus=min_focus,
        max_focus=max_focus,
        tol=tol,
        number_of_intervals=number_of_intervals,
        plot=plot,
        loop_img_processing_beamline=loop_img_processing_beamline,
        loop_img_processing_zoom=loop_img_processing_zoom,
        number_of_omega_steps=number_of_omega_steps,
        beam_size=beam_size,
        metadata=metadata,
        exposure_time=exposure_time,
        threshold=threshold,
    )
    
    yield from monitor_during_wrapper( 
        run_wrapper(_optical_and_xray_centering.start(), md=metadata), 
        signals=(sample_x, sample_y, alignment_x, alignment_y, alignment_z,
        omega, phase, backlight, _optical_and_xray_centering.grid_scan_coordinates_edge,
        _optical_and_xray_centering.grid_scan_coordinates_flat,
        _optical_and_xray_centering.md3_scan_response)
    )
