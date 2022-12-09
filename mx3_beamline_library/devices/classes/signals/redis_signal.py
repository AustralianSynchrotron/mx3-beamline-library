import struct
import time
import numpy as np
from typing import TYPE_CHECKING, Any, Union, NoReturn, Optional, Callable
from enum import Enum
from PIL import Image
from ophyd import DerivedSignal
from ophyd.utils import ReadOnlyError
from redis.exceptions import ConnectionError
from as_redis_signal.redis_signal import RedisSignal, NoKey

if TYPE_CHECKING:
    from redis import Redis
    from ophyd import Device
    from numpy.typing import NDArray
    from redis.client import PubSubWorkerThread, PubSub


class VideoModeMap(str, Enum):
    """Video Mode Mapping"""

    Mono8 = "L"
    RGB8Packed = "RGB"


class RedisSignalMD(RedisSignal):
    """Redis MD3 Signal"""

    def __init__(self,
        key: str,
        *args,
        r: Optional[Union["Redis", dict]] = None,
        serializer_deserializer: Optional[tuple[Callable, Callable]] = None,
        parent: Optional["Device"] = None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        key : str
            The redis key for this signal.
        r : Optional[Union[Redis, dict]], optional
            The redis instance or parameters, by default None
        serializer_deserializer : Optional[tuple[Callable, Callable]], optional
            A pair of serializer/deserializer callables, by default None
        parent : Optional[Device], optional
            The parent Device holding this signal, by default None

        Returns
        -------
        None
        """
        if parent is not None and parent._r is not None:
            r=parent._r
        super().__init__(
            key,
            *args,
            r=r,
            serializer_deserializer=serializer_deserializer,
            parent=parent,
            **kwargs
        )

    def read(self) -> dict[str, Any]:
        """Read stored value

        Returns
        -------
        dict[str, Any]
            A dictionary with name as the key, and the deserialized value as the value.

        Raises
        ------
        NoKey
            If the key is undefined.
        """
        v = self._r.get(self._key)
        if v is None:
            raise NoKey

        self._last_read = {"value": self._deserializer(v), "timestamp": time.time()}

        return {
            self.name: self._last_read,
        }

    def set(self, *args, **kwargs) -> NoReturn:
        "Disabled for a read-only signal"
        raise ReadOnlyError(f"The signal {self.name} is readonly.")


    def put(self, *args, **kwargs) -> NoReturn:
        "Disabled for a read-only signal"
        raise ReadOnlyError(f"The signal {self.name} is readonly.")

    @property
    def write_access(self) -> bool:
        "Can the signal be written to?"
        return False

    @property
    def metadata(self) -> dict:
        "A copy of the metadata dictionary associated with the signal"
        return {}

    def _callback(self, *args, **kwargs) -> None:
        """Subscription callback method.

        Returns
        -------
        None
        """
        read = self.read()[self.name]
        self._run_subs(
            sub_type=self.SUB_VALUE,
            old_value=self._last_read["value"],
            value=read["value"],
            timestamp=read["timestamp"],
        )


class RedisSignalMDImage(RedisSignalMD):
    """Redis MD3 Image Signal"""

    def __init__(
        self,
        key: str,
        *args,
        r: Optional[Union["Redis", dict]] = None,
        serializer_deserializer: Optional[tuple[Callable, Callable]] = None,
        parent: Optional["Device"] = None,
        mode_key: str = "video_mode",
        header_format: str = "<HiiHHQH",
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        key : str
            The redis key for this signal.
        r : Optional[Union[Redis, dict]], optional
            The redis instance or parameters, by default None
        serializer_deserializer : Optional[tuple[Callable, Callable]], optional
            A pair of serializer/deserializer callables, by default None
        parent : Optional[Device], optional
            The parent Device holding this signal, by default None
        mode_key : str, optional
            Key to read the current video mode from, by default "video_mode"
        header_format : str, optional
            Header format to parse the raw image data, by default "<HiiHHQH"

        Returns
        -------
        None
        """
        self._mode_key = mode_key
        self._header_format = header_format
        self._header_size = struct.calcsize(header_format)
        self._subscription_thread: Union["PubSubWorkerThread", None]
        if serializer_deserializer is None:
            serializer_deserializer = (None, self.__img_deserializer)
        super().__init__(
            key,
            *args,
            r=r,
            serializer_deserializer=serializer_deserializer,
            parent=parent,
            **kwargs
        )

    def __img_deserializer(self, value: bytes, mode: Union[bytes, str]) -> "NDArray":
        """MD Image deserializer

        Parameters
        ----------
        value : bytes
            Byte string consisting of header and raw image data.
        mode : Union[bytes, str]
            Current mode the MD video server is outputting in (Mono8, RGB8Packed).

        Returns
        -------
        NDArray
            Numpy array containing deserialized image data.
        """

        if isinstance(mode, bytes):
            mode = mode.decode("UTF-8")

        _, width, height, _, _, _, _ = struct.unpack(
            self._header_format, value[:self._header_size]
        )
        raw_image = value[self._header_size:]

        return np.array(
            Image.frombytes(
                mode=VideoModeMap[mode], size=(width, height), data=raw_image
            ),
        )

    def get(self) -> "NDArray":
        """Get value stored in redis

        Returns
        -------
        NDArray
            Numpy array containing deserialized image data.
        """
        return super().get()

    def read(self) -> dict[str, Any]:
        """Read stored value

        Returns
        -------
        dict[str, Any]
            A dictionary with name as the key, and the deserialized value as the value.

        Raises
        ------
        NoKey
            If either the raw image or mode key is undefined.
        """
        value, mode = self._r.mget([self._key, self._mode_key])

        if value is None or mode is None:
            raise NoKey

        self._last_read = {
            "value": self._deserializer(value=value, mode=mode),
            "timestamp": time.time(),
        }

        return {
            self.name: self._last_read,
        }

    def subscribe(self, *args, **kwargs) -> Union[int, None]:
        """Subscribe to redis signal. If key is updated in redis, subscription
        callback(s) will be fired.

        Handles:

            * Starting subscription thread if not already running.

        Returns
        -------
        Union[int, None]
            Id of callback, can be passed to `unsubscribe` to remove the callback,
            None if thread is already running and alive.
        """

        def exception_handler(e: Exception, pubsub: "PubSub", thread: "PubSubWorkerThread"):
            if not isinstance(e, ConnectionError):
                raise e
            thread.stop()
            pubsub.close()

        if self._pubsub is None:
            self._pubsub = self._r.pubsub(ignore_subscribe_messages=True)

        self._pubsub.subscribe(**{self._key: self._callback})
        if self._subscription_thread is not None:
            if self._subscription_thread.is_alive():
                return None
        self._subscription_thread = self._pubsub.run_in_thread(
            sleep_time=None, daemon=True, exception_handler=exception_handler
        )

        cid = super(RedisSignal, self).subscribe(*args, **kwargs)

        return cid

    def _delete_subscription(self) -> None:
        """Unsubscribe from key and stop the subscription thread.

        Returns
        -------
        None
        """
        self._pubsub.unsubscribe(self._key)
        self._subscription_thread.stop()


class MDDerivedDepth(DerivedSignal):
    """MD3 Derived Depth Signal"""

    def set(self, *args, **kwargs) -> NoReturn:
        "Disabled for a read-only signal"
        raise ReadOnlyError(f"The signal {self.name} is readonly.")

    def put(self, *args, **kwargs) -> NoReturn:
        "Disabled for a read-only signal"
        raise ReadOnlyError(f"The signal {self.name} is readonly.")

    def inverse(self, value: "NDArray") -> int:
        """Compute original signal value -> derived signal value.

        Parameters
        ----------
        value : NDArray
            Numpy array containing deserialized image data.

        Returns
        -------
        int
            Depth/Shape of the array.
        """
        if isinstance(value, np.ndarray) and len(value.shape) == 3:
            value.shape[2]
        return -1
