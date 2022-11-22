from __future__ import annotations

import os
import typing
import requests
import os.path
import functools
import wrapt
from typing import Any, Dict, Union, Optional, Generator
from requests_mock import Mocker as RequestMocker
from pydantic import BaseModel, Field
from ....classes.detectors import DectrisDetector

if typing.TYPE_CHECKING:
    from requests_mock.request import _RequestObjectProxy as Request
    from requests_mock.response import _Context as Context
    from functools import partial


ENABLE_DECTRIS_MOCKER = os.environ.get("ENABLE_DECTRIS_MOCKER", "True").lower() == "true"


class DectrisConfig(BaseModel):
    """Dectris Config SubModel"""

    sensor_thickness: float = Field(title="Sensor Thickness", default=0.000450)
    sensor_material: str = Field(title="Sensor Material", default="Si")
    x_pixel_size: float = Field(title="Pixel Size (X)", default=0.000075000000000)
    y_pixel_size: float = Field(title="Pixel Size (Y)", default=0.000075000000000)
    auto_summation: bool = Field(title="Auto Summation", default=True)
    eiger_fw_version: str = Field(title="Firmware Version", default="release-2020.2.5")
    software_version: str = Field(title="Software Version", default="1.8.0")
    description: str = Field(title="Description", default="Dectris EIGER2 Si 9M")
    detector_number: str = Field(title="Model Number", default="E-18-0108")
    detector_type: str = Field(title="Detector Type", default="HPC")
    detector_readout_time: float = Field(title="Readout Time", default=0.0000001)
    bit_depth_image: int = Field(title="Bit Depth (Image)", default=32)
    bit_depth_readout: int = Field(title="Bit Depth (Readout)", default=16)
    compression: str = Field(title="Compression Algorithm", default="bslz4")
    countrate_correction_count_cutoff: int = Field(
        title="Countrate Correction Cutoff",
        default=133343,
    )
    frame_count_time: float = Field(
        title="Frame Count Time",
        default=0.004170816650000,
    )
    number_of_excluded_pixels: int = Field(
        title="Number of Excluded Pixels",
        default=664708,
    )
    trigger_mode: str = Field(title="Trigger Mode", default="exts")
    x_pixels_in_detector: str = Field(title="Detector Pixels (X)", default=3108)
    y_pixels_in_detector: str = Field(title="Detector Pixels (Y)", default=3262)
    frame_time: int = Field(title="Frame Time", default=0)
    nimages: int = Field(title="Image Number", default=0)

    class Config:
        validate_assignment = True


class DectrisState(BaseModel):
    """Dectris State Model"""

    config: DectrisConfig = Field(
        title="Config Parameters",
        exclude=True,
        default_factory=DectrisConfig,
    )
    sequence_id: int = Field(
        title="Sequence ID",
        default=0,
    )
    image_number: int = Field(
        title="Image Number",
        default=0,
    )


class DectrisMocker:
    """Dectris API Mocker.

    Mocks the functionality of the Dectris detector API endpoints.
    """

    def __init__(
        self,
        rest: Optional[str] = None,
        base: str = "detector/api/1.8.0",
        cls: object = DectrisDetector,
    ) -> None:
        """Dectris API Mocker.

        Mocks the functionality of the Dectris detector API endpoints.

        Parameters
        ----------
        rest : Optional[str], optional
            REST API domain to simulate, by default None
        base : str, optional
            Base API endpoint, by default "detector/api/1.8.0"
        cls : object, optional
            Ophyd detector class to be mocked, by default DectrisDetector
        """

        self._rest = rest
        self._base = base
        self._cls = cls
        self._state = DectrisState()
        self._mocker = None

        if self._rest is not None and ENABLE_DECTRIS_MOCKER:
            self._mocker = self.build_mocker()

    @wrapt.decorator(enabled=ENABLE_DECTRIS_MOCKER)
    def __call__(self, obj: Union[type, "function"], instance: Any, args: tuple, kwargs: dict) -> Any:
        """_summary_

        Parameters
        ----------
        obj : Union[type, function]
            Object to be decorated, either a class or method.
        instance : Any
            Instance of object if called for an instantiated class.
        args : tuple
            Arguments to pass to decorated object.
        kwargs : dict
            Kwargs to pass to decorated object.

        Returns
        -------
        Any
            New wrapped class instance.
        """

        if self._rest is None:
            self._rest = kwargs.get("REST", (len(args) and args[0] or None))

        if self._mocker is None:
            self._mocker = self.build_mocker()

        if isinstance(obj, type):
            return self.decorate_class(obj, *args, **kwargs)

        return self.decorate_callable(obj)

    def get_cls_methods(self) -> Generator[str, None, None]:
        """Get class methods for decorated class.

        Yields
        ------
        Generator[str, None, None]
            Generator of named of callable attributes.
        """

        def _attr_is_dec_method(attr_name: str) -> bool:
            if attr_name.startswith("_"):
                return False
            attr = getattr(self._cls, attr_name, None)
            return callable(attr)

        for attr_name in filter(_attr_is_dec_method, self._cls.__dict__.keys()):
            yield attr_name

    def decorate_callable(self, func: "function") -> Union["partial", Any]:
        """Decorate callable.

        Wraps method to mock Dectris detector endpoints for REST API calls.

        Parameters
        ----------
        func : Optional[function], optional
            Method to be wrapped, by default None

        Returns
        -------
        Union[partial, Any]
            Partial for initial call when decorating with brackets "@decorator()",
            the next call (or initial if not decorating with brackets) will return
            the return value of the wrapped method.
        """

        if func is None:
            return functools.partial(self.decorate_callable)

        @wrapt.decorator(enabled=ENABLE_DECTRIS_MOCKER)
        def wrapper(func: "function", instance: Any, args: tuple, kwargs: dict) -> Any:
            with self._mocker:
                return func(*args, **kwargs)

        return wrapper(func)

    def decorate_class(self, cls: type, *args, **kwargs) -> Any:
        """Decorate class.

        Wraps class methods to mock Dectris detector endpoints for REST API calls.

        Parameters
        ----------
        cls : type
            Class being decorated.

        Returns
        -------
        Any
            New instance of class being decorated.
        """

        # Decorate class methods
        for attr_name in self.get_cls_methods():
            if not hasattr(cls, attr_name):
                continue

            attr = getattr(cls, attr_name)
            setattr(cls, attr_name, self.decorate_callable(attr))

        # Add methods to call mock endpoints
        for attr_name in ["get", "post", "put", "patch", "options", "delete"]:
            attr = getattr(requests, attr_name)
            setattr(cls, f"mock_{attr_name}", self.decorate_callable(staticmethod(attr)))

        return cls(*args, **kwargs)

    def build_mocker(self) -> RequestMocker:
        """Build request mocker.

        Builds and configures request mocker instance to
        simulate Dectris REST endpoints.

        Returns
        -------
        RequestMocker
            Built instance of RequestMocker.
        """

        def rest_callback(func: Optional["function"] = None, state: DectrisState = self._state) -> Union["partial", Any]:
            """Dectris REST callback wrapper.

            Wraps Dectris REST callback methods used by the "rest_mocker" package.
            Wrapper passes the Dectris detector state object to the callback method.

            Parameters
            ----------
            func : Optional[function], optional
                Method to be wrapped, by default None
            state : DectrisState, optional
                Dectris detector state object, by default self._state

            Returns
            -------
            Union[partial, Any]
                Partial for initial call when decorating with brackets "@decorator()",
                the next call (or initial if not decorating with brackets) will return
                the return value of the wrapped method.
            """

            if func is None:
                return functools.partial(rest_callback, state=state)

            @wrapt.decorator(enabled=ENABLE_DECTRIS_MOCKER)
            def wrapper(func: "function", instance: Any, request: Union["Request", tuple["Request", "Context"]], context: Optional[Union["Context", dict]] = None) -> Any:
                if isinstance(request, tuple):
                    request, context = request
                return func(request, context, state)

            return wrapper(func)


        @rest_callback
        def mocked_config_put(request: "Request", context: "Context", state: DectrisState) -> Dict[str, Any]:
            """Mocks Dectris REST API endpoint "PUT" operations for config.

            Parameters
            ----------
            request : Request
                HTTP request object.
            context : Context
                Request context, contains cookies, etc.
            state : DectrisState
                Object to maintain the mocked Dectris state.

            Returns
            -------
            Dict[str, Any]
                JSON serialisable object to be returned in response body.
            """

            endpoint = os.path.split(request.path)[-1]
            request_dict: dict = request.json()
            setattr(state.config, endpoint, request_dict.get("value"))

            return {"value": getattr(state.config, endpoint)}


        @rest_callback
        def mocked_config_get(request: "Request", context: "Context", state: DectrisState) -> Dict[str, Any]:
            """Mocks Dectris REST API endpoint "GET" operations for config.

            Parameters
            ----------
            request : Request
                HTTP request object.
            context : Context
                Request context, contains cookies, etc.
            state : DectrisState
                Object to maintain the mocked Dectris state.

            Returns
            -------
            Dict[str, Any]
                JSON serialisable object to be returned in response body.
            """

            endpoint = os.path.split(request.path)[-1]

            return {"value": getattr(state.config, endpoint)}


        @rest_callback
        def metadata_get(request: "Request", context: "Context", state: DectrisState) -> Dict[str, Any]:
            """Dectris REST API endpoint to return metadata for testing.

            Parameters
            ----------
            request : Request
                HTTP request object.
            context : Context
                Request context, contains cookies, etc.
            state : DectrisState
                Object to maintain the mocked Dectris state.

            Returns
            -------
            Dict[str, Any]
                JSON serialisable object to be returned in response body.
            """

            return {"metadata": state.dict()}

        @rest_callback
        def mocked_arm(request: "Request", context: "Context", state: DectrisState) -> Dict[str, Any]:
            """Mocks the Dectris REST API endpoint for arming the detector.

            Parameters
            ----------
            request : Request
                HTTP request object.
            context : Context
                Request context, contains cookies, etc.
            state : DectrisState
                Object to maintain the mocked Dectris state.

            Returns
            -------
            Dict[str, Any]
                JSON serialisable object to be returned in response body.
            """

            # Increment sequence_id
            state.sequence_id += 1

            # Reset the image number every time we arm the detector
            state.image_number = 0

            return {"sequence id": state.sequence_id}

        mocker = RequestMocker()

        # Mock root path endpoint
        mocker.get(f"{self._rest}/", json={"SIMplonAPI": "Mocked"})

        # Mock detector config endpoint (GET)
        for endpoint in self._state.config.dict().keys():
            mocker.get(
                os.path.join(self._rest, self._base, "config", endpoint),
                json=mocked_config_get,
            )

        # Mock detector config (PUT)
        for endpoint in ("frame_time", "nimages"):
            mocker.put(
                os.path.join(self._rest, self._base, "config", endpoint),
                json=mocked_config_put,
            )

        # Mock trigger endpoint
        mocker.put(os.path.join(self._rest, self._base, "command/trigger"), status_code=200)

        # Mock arm endpoint
        mocker.put(
            os.path.join(self._rest, self._base, "command/arm"),
            json=mocked_arm,
        )

        # Mock disarm endpoint
        mocker.put(os.path.join(self._rest, self._base, "command/disarm"), status_code=200)

        # Add mocker metadata endpoint for tests
        mocker.get(os.path.join(self._rest, "metadata"), json=metadata_get)

        return mocker
