""" Beamline detector definition """

import logging
import os
import struct
from os import environ
from typing import TYPE_CHECKING, Any, Optional, Union

import bitshuffle
import h5py
import hdf5plugin  # noqa
import requests
from ophyd import ADComponent, AreaDetector, Component as Cpt, Device, cam
from ophyd.areadetector.plugins import (
    ColorConvPlugin,
    ImagePlugin_V25 as ImagePlugin,
    StatsPlugin_V33,
)
from ophyd.signal import EpicsSignal, EpicsSignalRO, Signal
from ophyd.status import Status

from ...config import OPTICAL_CENTERING_CONFIG
from . import Register

logger = logging.getLogger(__name__)
_stream_handler = logging.StreamHandler()
logging.getLogger(__name__).addHandler(_stream_handler)
logging.getLogger(__name__).setLevel(logging.INFO)


try:
    from .signals.redis_signal import MDDerivedDepth, RedisSignalMD, RedisSignalMDImage
except NameError:
    RedisSignalMD = Signal
    MDDerivedDepth = Signal
    RedisSignalMDImage = Signal


if TYPE_CHECKING:
    from redis import Redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

HDF5_OUTPUT_DIRECTORY = environ.get("HDF5_OUTPUT_DIRECTORY", os.getcwd())


class HDF5Filewriter(ImagePlugin):
    filename = Cpt(Signal, name="filename", kind="config", value="hdf5_file.h5")
    image_id = Cpt(Signal, name="image_id", kind="hinted")
    write_path_template = Cpt(
        Signal, name="write_path_template", kind="config", value=HDF5_OUTPUT_DIRECTORY
    )
    compression = Cpt(Signal, name="compression", kind="config", value="bslz4")
    frames_per_datafile = Cpt(
        Signal, name="frames_per_datafile", kind="omitted", value=1
    )
    # _default_read_attrs = ("image_id", "compression", "write_path_template")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._datafile = None
        self._image_id = None
        self._height = None
        self._width = None
        self.hdf5_path = None

    def stage(self) -> None:
        """
        Opens an hdf5 datafile and sets all the information needed to start
        writing files to disk

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raises an error if frames_per_datafile has not been set
        """
        self.write_path_template.set(self.write_path_template.get())
        self.filename.set(self.filename.get())
        self.compression.set(self.compression.get())
        if self.frames_per_datafile.get() is None:
            raise ValueError("frames_per_datafile has not been set")

        self._image_id = 0
        self._width = self.width.get()
        self._height = self.height.get()
        self._dtype = str(self.image.dtype)

    def trigger(self) -> Status:
        """
        Gets a frame from the image array plugin, and saves the frame to disk

        Returns
        -------
        Status
            the status of the device

        Raises
        ------
        NotImplementedError
            Raises an error if the data type of the array is not
            uint8, uint16, or uint32
        """
        if self._datafile is None:
            self._datafile = self._create_empty_datafile(
                single_img_shape=(self._height, self._width),
                dtype=self._dtype,
                frames_per_datafile=self.frames_per_datafile.get(),
                compression=self.compression.get(),
            )
        self.image_id.set(self._image_id)
        d = Status(self)
        if self.compression.get() is None:
            image_array = self.image
        elif self.compression.get() == "bslz4":
            image = bitshuffle.compress_lz4(self.image).tobytes()
            # NOTE: The byte number of elements depends on the data type
            if self._dtype == "uint32" or self._dtype == "int32":
                element_size = 4
            elif self._dtype == "uint16" or self._dtype == "int16":
                element_size = 2
            elif self._dtype == "uint8" or self._dtype == "int8":
                element_size = 1
            else:
                raise NotImplementedError(
                    "Supported image types are uint32, uint16, uint8, int32, int16, "
                    f"and int8, not {self._dtype}"
                )
            bytes_number_of_elements = struct.pack(
                ">q", (self._width * self._height * element_size)
            )
            bytes_block_size = struct.pack(">I", 0)
            image_array = bytes_number_of_elements + bytes_block_size + image

        self._write_direct_chunks(
            image=image_array, image_id=self._image_id, hdf5_file=self._datafile
        )
        self._image_id += 1
        d._finished()
        return d

    def unstage(self) -> None:
        """
        Closes the HDF5 Fle

        Returns
        -------
        None
        """
        if self._datafile is None:
            if self.hdf5_path is not None:
                if os.path.isfile(self.hdf5_path):
                    raise FileExistsError(
                        f"{self.hdf5_path} already exists. Choose a different file name"
                    )
            return
        self._datafile.close()
        self._datafile = None
        self._image_id = None
        self._height = None
        self._width = None

    def _generate_master_file_path(self) -> str:
        """
        Generates the master file path

        Returns
        -------
        str
            The master file path
        """
        _hdf5_path = os.path.join(self.write_path_template.get(), self.filename.get())
        file_name, file_extension = os.path.splitext(_hdf5_path)

        if len(file_extension) == 0:
            return file_name + ".h5"

        if file_extension != ".h5":
            logger.warning(
                "HDF5 filename extension does not end with `.h5`. Extension will "
                "be renamed to .h5. Final HDF5 filename: "
                f"{file_name + '.h5'}"
            )
            return file_name + ".h5"

        return _hdf5_path

    def _create_empty_datafile(
        self,
        single_img_shape: tuple[int, int],
        dtype: str,
        frames_per_datafile: int,
        compression: Union[str, None] = "bslz4",
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
            Compression type. Can be either bslz4, or None

        Returns
        -------
        hf : h5py.File
            An hdf5 file
            NOTE: This file has to be closed when all chunks of data have
            been written to disk to avoid memory leaks
        """
        self.hdf5_path = self._generate_master_file_path()

        hf = h5py.File(self.hdf5_path, "w-")

        # entry/data (group)
        data = hf.create_group("entry/data")

        shape = (frames_per_datafile, single_img_shape[0], single_img_shape[1])
        chunks = (1, single_img_shape[0], single_img_shape[1])

        # entry/data/data (dataset)
        if compression == "bslz4":
            # NOTE: the compression ids (a.k.a filter_id) below come from
            # https://www.silx.org/doc/hdf5plugin/latest/usage.htmls
            data.create_dataset(
                "frames",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compression=32008,
                compression_opts=(0, 2),
            )
        elif compression == "lz4":
            data.create_dataset(
                "frames",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compression=32004,
            )

        elif compression is None:
            data.create_dataset(
                "frames",
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
        frame_id = image_id % self.frames_per_datafile.get()
        hdf5_file["entry/data/frames"].id.write_direct_chunk((frame_id, 0, 0), image, 0)


@Register("Blackfly Camera")
class BlackflyCamera(AreaDetector):
    image = ADComponent(ImagePlugin, ":" + ImagePlugin._default_suffix, lazy=True)
    cam = ADComponent(cam.AreaDetectorCam, ":cam1:", lazy=True)
    color_plugin = ADComponent(ColorConvPlugin, ":CC1:", lazy=True)
    stats = ADComponent(
        StatsPlugin_V33, ":" + StatsPlugin_V33._default_suffix, kind="hinted", lazy=True
    )
    hdf5_filewriter = ADComponent(
        HDF5Filewriter, ":" + ImagePlugin._default_suffix, lazy=True
    )


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

    pixels_per_mm_x = OPTICAL_CENTERING_CONFIG.top_camera.pixels_per_mm_x
    pixels_per_mm_y = OPTICAL_CENTERING_CONFIG.top_camera.pixels_per_mm_y


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
