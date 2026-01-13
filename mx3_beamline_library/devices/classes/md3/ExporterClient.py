import socket
import time
from typing import Any, Iterable

from pydantic import BaseModel


class ExporterProtocolError(Exception):
    pass


class ExporterTimeoutError(TimeoutError):
    pass


STX = chr(2)
ETX = chr(3)

CMD_METHOD_LIST = "LIST"
CMD_PROPERTY_READ = "READ"
CMD_PROPERTY_WRITE = "WRTE"
CMD_PROPERTY_LIST = "PLST"
CMD_NAME = "NAME"
CMD_SYNC_CALL = "EXEC"
CMD_ASNC_CALL = "ASNC"

RET_ERR = "ERR:"
RET_OK = "RET:"
RET_NULL = "NULL"

EVENT = "EVT:"

PARAMETER_SEPARATOR = "\t"
ARRAY_SEPARATOR = "\x1f"  # 0x001F (previously rendered as "")


class ExporterAddress(BaseModel):
    host: str
    port: int


class ExporterClient:
    """
    Synchronous Exporter client without threading
    """

    def __init__(self, address: str, port: int, timeout: float = 3.0):
        self.address = ExporterAddress(host=address, port=int(port))
        self.timeout = timeout

    def _send_receive(self, payload: str) -> str:
        """
        Send a request and wait for a reply.

        Parameters
        ----------
        payload : str
            The request payload (without STX/ETX).

        Returns
        -------
        str
            The reply payload (without STX/ETX).
        """
        deadline = time.monotonic() + self.timeout
        # TODO: is this timeout long enough if we trigger collections?
        with socket.create_connection(
            (self.address.host, self.address.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            sock.sendall((STX + payload + ETX).encode())

            buf: list[str] = []
            in_frame = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ExporterTimeoutError(
                        f"Timeout waiting for reply from {self.address.host}:{self.address.port}"
                    )
                sock.settimeout(min(self.timeout, remaining))

                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed by peer")
                for ch in chunk.decode(errors="replace"):
                    if ch == STX:
                        buf.clear()
                        in_frame = True
                    elif ch == ETX:
                        if in_frame:
                            msg = "".join(buf)
                            if msg.startswith(EVENT):
                                self._handle_event(msg)
                                buf.clear()
                                in_frame = False
                                continue
                            return msg
                        # ignore stray ETX
                    else:
                        if in_frame:
                            buf.append(ch)

    def _send_only(self, payload: str) -> None:
        """
        Send a request without waiting for a reply.

        Parameters
        ----------
        payload : str
            The request payload (without STX/ETX).

        Returns
        -------
        None
        """
        with socket.create_connection(
            (self.address.host, self.address.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            sock.sendall((STX + payload + ETX).encode())

    def _handle_event(self, msg: str) -> None:
        """
        Handle an incoming event message.
        Expected format: EVT:<name>\t<value>\t<timestamp>

        Parameters
        ----------
        msg : str
            The event message (including the EVT: prefix).

        Returns
        -------
        None
        """
        try:
            evtstr = msg[len(EVENT) :]
            tokens = evtstr.split(PARAMETER_SEPARATOR)
            if len(tokens) < 3:
                return
            self.onEvent(tokens[0], tokens[1], int(tokens[2]))
        except Exception:
            # Keep behaviour compatible with upstream: ignore event parsing errors.
            return

    def _process_return(self, ret: str) -> str | None:
        """
        Process the return value from a request.

        Parameters
        ----------
        ret : str
            The raw return string.

        Returns
        -------
        str | None
            The processed return value.

        Raises
        ------
        RuntimeError
            If the return value indicates an error.
        ExporterProtocolError
            If the return value is unexpected.
        """
        if ret.startswith(RET_ERR):
            raise RuntimeError(ret[len(RET_ERR) :])
        if ret == RET_NULL:
            return None
        if ret.startswith(RET_OK):
            return ret[len(RET_OK) :]
        raise ExporterProtocolError(f"Unexpected reply: {ret!r}")

    def _to_python_value(self, value: Any) -> Any:
        """
        Convert a string value to an appropriate Python data type.

        Parameters
        ----------
        value : Any
            The value to convert.
        Returns
        -------
        Any
            The converted Python value.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return value

        if ARRAY_SEPARATOR in value:
            parsed = self.parse_array(value)
            if parsed is None:
                return value
            # attempt numeric conversion
            try:
                return [int(x) for x in parsed]
            except Exception:
                try:
                    return [float(x) for x in parsed]
                except Exception:
                    return parsed

        lowered = value.lower()
        if lowered == "false":
            return False
        if lowered == "true":
            return True
        try:
            return int(value)
        except Exception:
            try:
                return float(value)
            except Exception:
                return value

    # ---- Public API (Exporter-like) ----
    # def getMethodList(self) -> list[str] | None:
    #     ret = self._process_return(self._send_receive(CMD_METHOD_LIST))
    #     if ret is None:
    #         return None
    #     parts = ret.split(PARAMETER_SEPARATOR)
    #     if len(parts) > 1 and parts[-1] == "":
    #         parts = parts[:-1]
    #     return parts

    # def getPropertyList(self) -> list[str] | None:
    #     ret = self._process_return(self._send_receive(CMD_PROPERTY_LIST))
    #     if ret is None:
    #         return None
    #     parts = ret.split(PARAMETER_SEPARATOR)
    #     if len(parts) > 1 and parts[-1] == "":
    #         parts = parts[:-1]
    #     return parts

    # def getServerObjectName(self) -> str | None:
    #     return self._process_return(self._send_receive(CMD_NAME))

    def execute(self, method: str, *pars: Any, timeout: float | None = None) -> Any:
        """
        Execute a synchronous method call

        Parameters
        ----------
        method : str
            The method name to call.
        *pars : Any
            The method parameters.
        timeout : float | None, optional
            The timeout for the call (not used currently), by default None
        """
        cmd = f"{CMD_SYNC_CALL} {method} "
        if pars:
            for par in pars:
                if isinstance(par, (list, tuple)):
                    par = self._create_array_parameter(par)
                cmd += f"{par}{PARAMETER_SEPARATOR}"
        ret = self._process_return(self._send_receive(cmd))
        return self._to_python_value(ret)

    def executeAsync(self, method: str, *pars: Any) -> None:
        """
        Execute an asynchronous method call. Currently not used
        by the beamline library. We simply send the ASNC request and return,
        no tracking of completion/result is done.

        Parameters
        ----------
        method : str
            The method name to call.
        *pars : Any
            The method parameters.

        Returns
        -------
        None
        """
        cmd = f"{CMD_ASNC_CALL} {method} "
        if pars:
            for par in pars:
                if isinstance(par, (list, tuple)):
                    par = self._create_array_parameter(par)
                cmd += f"{par}{PARAMETER_SEPARATOR}"
        self._send_only(cmd)

    def read_property(self, prop: str) -> Any:
        ret = self._process_return(self._send_receive(f"{CMD_PROPERTY_READ} {prop}"))
        return self._to_python_value(ret)

    # def readPropertyAsString(self, prop: str) -> str | None:
    #     val = self._process_return(self._send_receive(f"{CMD_PROPERTY_READ} {prop}"))
    #     return None if val is None else str(val)

    # def readPropertyAsFloat(self, prop: str) -> float:
    #     return float(self.readProperty(prop))

    # def readPropertyAsInt(self, prop: str) -> int:
    #     return int(self.readProperty(prop))

    # def readPropertyAsBoolean(self, prop: str) -> bool:
    #     return bool(self.readProperty(prop))

    # def readPropertyAsStringArray(self, prop: str) -> list[str] | None:
    #     ret = self._process_return(self._send_receive(f"{CMD_PROPERTY_READ} {prop}"))
    #     if ret is None:
    #         return None
    #     return self.parse_array(str(ret))

    def write_property(self, prop: str, value: Any) -> Any:
        """
        Write a property value.

        Parameters
        ----------
        prop : str
            The property name.
        value : Any
            The value to write.

        Returns
        -------
        Any
            The written value converted to Python type.
        """
        if isinstance(value, (list, tuple)):
            value = self._create_array_parameter(value)
        ret = self._process_return(
            self._send_receive(f"{CMD_PROPERTY_WRITE} {prop} {value}")
        )
        return self._to_python_value(ret)

    def parse_array(self, value: str) -> list[str] | None:
        """
        Parse a string representing an array into a list of strings.

        Parameters
        ----------
        value : str
            The string to parse.
        Returns
        -------
        list[str] | None
            The parsed list of strings, or None if the input is not an array.
        """
        value = str(value)
        if not value.startswith(ARRAY_SEPARATOR):
            return None
        if value == ARRAY_SEPARATOR:
            return []
        value = value.lstrip(ARRAY_SEPARATOR).rstrip(ARRAY_SEPARATOR)
        return value.split(ARRAY_SEPARATOR)

    def _create_array_parameter(self, value: Iterable[Any]) -> str:
        """
        Create a string representation of an array parameter.

        Parameters
        ----------
        value : Iterable[Any]
            The iterable of values to convert.

        Returns
        -------
        str
            The string representation of the array parameter.
        """
        return "".join(f"{item}{PARAMETER_SEPARATOR}" for item in value)

    # ---- Upstream API aliases (to ease transition) ----
    # def parseArray(self, value: str) -> list[str] | None:
    #     return self.parse_array(value)

    # def createArrayParameter(self, value: Iterable[Any]) -> str:
    #     return self._create_array_parameter(value)

    def onEvent(self, name: str, value: Any, timestamp: int) -> None:
        # TODO: implement event handling if needed
        return

    def retrieve_task_info(self, taskId: Any) -> Any:
        """
        Retrieve information about a task.

        Parameters
        ----------
        taskId : Any
            The task identifier.

        Returns
        -------
        Any
            The task information.
        """
        return self.execute("getTaskInfo", taskId)

    def moveAndWaitEndOfMove(
        self,
        motor: str,
        initialPos: float,
        position: float,
        useAttr: bool,
        n: int,
        useEvents: bool = False,
        goBack: bool = False,
        timeout: float = 20,
        backMove: bool = False,
    ) -> None:
        # Minimal, polling-based implementation (no events/threads).
        # `useEvents` is ignored.
        motor_name = motor.replace(" ", "")

        def wait_motor_ready(deadline: float) -> None:
            while True:
                state = str(self.read_property(f"{motor_name}State"))
                if state in {"Ready", "LowLim", "HighLim"}:
                    return
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timeout waiting for {motor} to become Ready (state={state})"
                    )
                time.sleep(0.1)

        def do_move(target: float, deadline: float) -> None:
            if useAttr:
                self.write_property(f"{motor_name}Position", target)
            else:
                # Prefer positional arguments (Exporter protocol).
                self.execute("setMotorPosition", motor, target)
            wait_motor_ready(deadline)

        # Ensure starting position if requested
        try:
            current = float(self.execute("getMotorPosition", motor))
        except Exception:
            current = float(initialPos)

        overall_deadline = time.monotonic() + float(timeout)
        if abs(current - float(initialPos)) > 0.01:
            do_move(float(initialPos), overall_deadline)

        for _ in range(int(n)):
            do_move(float(position), overall_deadline)
            if goBack and not backMove:
                do_move(float(initialPos), overall_deadline)

    # ---- Python conveniences ----
    def __getitem__(self, key: str) -> Any:
        return self.read_property(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.write_property(key, value)

    def __getattr__(self, name: str):
        # Provide a simple, dynamic method proxy:
        #   client.getState() -> execute("getState")
        #   client.setBeamPositionHorizontal(1.2) -> execute("setBeamPositionHorizontal", 1.2)
        def caller(*args: Any, **kwargs: Any):
            if kwargs:
                raise TypeError(
                    "ExporterClient methods only support positional arguments; "
                    f"got unexpected keyword args: {sorted(kwargs.keys())}"
                )
            return self.execute(name, *args)

        return caller
