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
import h5py
import os
if TYPE_CHECKING:
    from redis import Redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

class WriteFileSignal(EpicsSignalWithRBV):
    def trigger(self):
        self.set(1, timeout=0)
        d = Status(self)
        d._finished()
        return d
    
class HDF5Filewriter(ImagePlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames_per_datafile = None
        self._datafile = None
        self._image_id = None
        self._height = None
        self._width = None
    
    def stage(self):
        if self.frames_per_datafile is None:
            raise ValueError("frames_per_datafile has not been set")
        self._image_id = 0
        self._width = self.width.get()
        self._height = self.height.get()
        self._datafile = self._create_empty_datafile(single_img_shape=(self._height, self._width), dtype="uint8", frames_per_datafile=self.frames_per_datafile)

    def trigger(self):
        d = Status(self)
        image = self.array_data.get().reshape(self._height, self._width)
        self._write_direct_chunks(image=image,image_id=self._image_id, hdf5_file=self._datafile)
        self._image_id += 1
        d._finished()
        return d

    def unstage(self):
        self._datafile.close()
        self.frames_per_datafile = None
        self._datafile = None
        self._image_id = None
        self._height = None
        self._width = None
    
    def _create_empty_datafile(
        self,
        single_img_shape: tuple[int, int],
        dtype: str,
        frames_per_datafile: int,
        compression: None=None,
    ) -> h5py.File:
        """
        Creates an empty data file.

        Parameters
        ----------
        single_img_shape : tuple[int, int]
            Shape of an individual frame
        dtype : str
            Data type, e.g. 'uint8'
        frames_per_datafile : int
            Number of frames per datafile
        compression : str | None
            Compression type. Can be either bslz4, lz4, or None

        Returns
        -------
        hf : h5py.File
            An hdf5 file
            NOTE: This file has to be closed when all chunks of data have
            been written to disk to avoid memory leaks
        """
        filename = os.path.join(
            os.getcwd(), f"test.h5"
        )
        hf = h5py.File(filename, "w")

        # entry/data (group)
        data = hf.create_group("entry/data")
        data.attrs["NX_class"] = "NXdata"

        shape = (frames_per_datafile, single_img_shape[0], single_img_shape[1])
        chunks = (1, single_img_shape[0], single_img_shape[1])

        # entry/data/data (dataset)
        if compression == "bslz4":
            # NOTE: the compression ids (a.k.a filter_id) below come from
            # https://www.silx.org/doc/hdf5plugin/latest/usage.htmls
            data.create_dataset(
                "data",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compression=32008,
                compression_opts=(0, 2),
            )
        elif compression == "lz4":
            data.create_dataset(
                "data",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compression=32004,
            )

        elif compression is None:
            data.create_dataset(
                "data",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
            )
        else:
            raise ValueError(
                f"Compression {compression} not supported, allowed values are "
                "bslz4, lz4 or None"
            )

        return hf
    
    def _write_direct_chunks(
        self, image: bytes, image_id: int, hdf5_file: h5py.File
    ) -> None:
        """
        Writes direct chunks of frames into an hdf5 dataset

        Parameters
        ----------
        image : bytes
            Compressed frame obtained from the zmq stream2
        image_id : int
            Image_id obtained from the zmq stream2
        hdf5_file : h5py.File
            A HDF5 datafile

        Returns
        -------
        None
        """
        frame_id = image_id % self.frames_per_datafile
        hdf5_file["entry/data/data"].id.write_direct_chunk((frame_id, 0, 0), image, 0)

class BlackflyCamHDF5Plugin(HDF5Plugin, FileStoreIterativeWrite):
    write_file =  Cpt(WriteFileSignal, "WriteFile")

class CommissioningBlackflyCamera(AreaDetector):
    image = ADComponent(ImagePlugin, ":" + ImagePlugin._default_suffix)
    cam = ADComponent(cam.AreaDetectorCam, ":cam1:")
    color_plugin = ADComponent(ColorConvPlugin, ":CC1:")
    stats = ADComponent(StatsPlugin_V33, ":" + StatsPlugin_V33._default_suffix)
    file_plugin = ADComponent(BlackflyCamHDF5Plugin, suffix=':HDF1:', write_path_template="/tmp")
    hdf5_filewriter = ADComponent(HDF5Filewriter, ":" + ImagePlugin._default_suffix)


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
