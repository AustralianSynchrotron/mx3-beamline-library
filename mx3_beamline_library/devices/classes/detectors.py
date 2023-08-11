""" Beamline detector definition """

import logging
from os import path
from typing import TYPE_CHECKING, Any, Optional, Union

import requests
import yaml
from ophyd import Component as Cpt, Device, AreaDetector, EpicsSignalWithRBV, ADComponent, cam, ImagePlugin
from ophyd.areadetector.plugins import ColorConvPlugin, StatsPlugin_V33
from ophyd.signal import EpicsSignal, EpicsSignalRO, Signal
from ophyd.status import Status
from ophyd.areadetector.plugins import HDF5Plugin
from ophyd.areadetector.filestore_mixins import FileStoreIterativeWrite

from .signals.redis_signal import MDDerivedDepth, RedisSignalMD, RedisSignalMDImage

if TYPE_CHECKING:
    from redis import Redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

class WriteFileSignal(EpicsSignalWithRBV):
    def trigger(self):
        self.set(1)
        d = Status(self)
        d._finished()
        return d

class BlackflyCamHDF5Plugin(HDF5Plugin, FileStoreIterativeWrite):
    write_file =  Cpt(WriteFileSignal, "WriteFile")

class CommissioningBlackflyCamera(AreaDetector):
    image = ADComponent(ImagePlugin, ":" + ImagePlugin._default_suffix)
    cam = ADComponent(cam.AreaDetectorCam, ":cam1:")
    color_plugin = ADComponent(ColorConvPlugin, ":CC1:")
    stats = ADComponent(StatsPlugin_V33, ":" + StatsPlugin_V33._default_suffix)
    file_plugin = ADComponent(BlackflyCamHDF5Plugin, suffix=':HDF1:', write_path_template="/tmp")


class BlackFlyCam(Device):
    """
    Ophyd device to acquire images from a Blackfly camera.

    Attributes
    ----------
    depth: float
        Depth of the camera image
    width: float
        Width of the camera image
    height: float
        Height of the camera image
    array_data : numpy array
        Array data
    acquire_time_rbv: int
        Acquire time of the camera image. Read-only
    gain_rbv: float
        Gain of the camera
    gain_auto: float
        Auto-gain of the camera
    gain_auto_rb: float
        Auto-gain of the camera. Read-only.
    frame_rate: int
        Frame rate of the camera images.
    """

    width = Cpt(EpicsSignalRO, ":image1:ArraySize0_RBV")
    height = Cpt(EpicsSignalRO, ":image1:ArraySize1_RBV")
    depth = Cpt(EpicsSignalRO, ":image1:ArraySize2_RBV")
    array_data = Cpt(EpicsSignalRO, ":image1:ArrayData")

    acquire_time_rbv = Cpt(EpicsSignalRO, ":cam1:AcquireTime_RBV")
    gain_rbv = Cpt(EpicsSignal, ":cam1:Gain_RBV")
    gain_auto = Cpt(EpicsSignal, ":cam1:GainAuto")
    gain_auto_rbv = Cpt(EpicsSignalRO, ":cam1:GainAuto_RBV")
    frame_rate = Cpt(EpicsSignal, ":cam1:FrameRate")

    path_to_config_file = path.join(
        path.dirname(__file__),
        "../../plans/configuration/optical_and_xray_centering.yml",
    )
    with open(path_to_config_file, "r") as plan_config:
        plan_args: dict = yaml.safe_load(plan_config)

    pixels_per_mm_x = plan_args["top_camera"]["pixels_per_mm_x"]
    pixels_per_mm_y = plan_args["top_camera"]["pixels_per_mm_y"]


class DectrisDetector(Device):
    """
    Signal wrapper used to call the Simplon API
    """

    sequence_id = Cpt(Signal, kind="hinted", name="sequence_id")

    def __init__(self, REST: str, *args, **kwargs) -> None:
        """
        Parameters
        ----------
        REST : str
            URL address

        Returns
        -------
        None
        """
        super().__init__(*args, **kwargs)
        self.REST = REST

    def configure(self, detector_configuration: dict) -> tuple[dict, dict]:
        """Configure the detector during a run

        Parameters
        ----------
        detector_configuration : dict
            The configuration dictionary. To specify the order that
            the changes should be made, use an OrderedDict.

        Returns
        -------
        (old_config, new_config) : tuple of dictionaries
            Where old and new are pre- and post-configure configuration states.
        """

        logging.info("Configuring detector...")

        new_config = detector_configuration

        for key, value in new_config.items():
            dict_data = {"value": value}

            if key == "user_data":
                # NOTE: the user_data is a special case because it references
                # the header_appendix endpoint, i.e. the key and the
                # endpoint don't match in this case
                r = requests.put(
                    f"{self.REST}/stream/api/1.8.0/config/header_appendix",
                    json=dict_data,
                )
            else:
                r = requests.put(
                    f"{self.REST}/detector/api/1.8.0/config/{key}", json=dict_data
                )
            if r.status_code == 200:
                logging.info(f"{key} set to {value}")
            else:
                logging.info(f"Could not set {key} to {value}")

        # Not implemented yet
        old_config = new_config

        return old_config, new_config

    def stage(self) -> None:
        """Arm detector

        Returns
        -------
        None
        """
        r = requests.put(f"{self.REST}/detector/api/1.8.0/command/arm")
        logging.info(
            f"arm: {r.json()}",
        )

        self.sequence_id.put(r.json()["sequence id"])

    def trigger(self) -> Status:
        """Trigger detector

        Returns
        -------
        d : Status
            Status of the detector
        """
        logging.info("Triggering detector...")
        r = requests.put(f"{self.REST}/detector/api/1.8.0/command/trigger")
        logging.info(f"trigger: {r.text}")

        d = Status(self)
        d._finished()
        return d

    def unstage(self) -> None:
        """Disarm detector

        Returns
        -------
        None
        """
        r = requests.put(f"{self.REST}/detector/api/1.8.0/command/disarm")
        logging.info(f"disarm: {r.text}")


class MDRedisCam(Device):
    """MD3 Redis Camera Device"""

    width = Cpt(RedisSignalMD, "image_width", lazy=True)
    height = Cpt(RedisSignalMD, "image_height", lazy=True)
    array_data = Cpt(RedisSignalMDImage, "bzoom:RAW", name="array_data", lazy=True)
    depth = Cpt(
        MDDerivedDepth, derived_from="array_data", write_access=False, lazy=True
    )

    acquire_time_rbv = Cpt(RedisSignalMD, "acquisition_frame_rate", lazy=True)
    frame_rate = Cpt(RedisSignalMD, "video_fps", lazy=True)

    def __init__(
        self,
        r: Optional[Union["Redis", dict]],
        *args,
        name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        r : Optional[Union[Redis, dict]]
            The redis instance or parameters.
        name : Optional[Any], optional
            Name of the device, by default None

        Returns
        -------
        None
        """
        self._r = r
        super().__init__(prefix="", *args, name=name, **kwargs)
