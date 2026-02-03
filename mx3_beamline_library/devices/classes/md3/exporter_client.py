import socket
import time
from typing import Any, Iterable

from pydantic import BaseModel

from . import logger


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
    Exporter client without threading
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
        with socket.create_connection(
            (self.address.host, self.address.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            sock.sendall((STX + payload + ETX).encode())

            buffer: list[str] = []
            in_message = False  # Determines if we are inside a message
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ExporterTimeoutError(
                        f"Timeout waiting for reply " f"from {self.address}"
                    )
                sock.settimeout(min(self.timeout, remaining))

                response = sock.recv(4096)
                if not response:
                    raise ConnectionError("Connection closed by peer")
                for res in response.decode(errors="replace"):
                    if res == STX:
                        buffer.clear()
                        in_message = True
                    elif res == ETX:
                        if in_message:
                            msg = "".join(buffer)
                            if msg.startswith(EVENT):
                                self._handle_event(msg)
                                buffer.clear()
                                in_message = False
                                continue
                            return msg
                    else:
                        if in_message:
                            buffer.append(res)

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
        Handle an event message.
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
            evt_str = msg[len(EVENT) :]
            tokens = evt_str.split(PARAMETER_SEPARATOR)
            if len(tokens) < 3:
                return
            self.on_event(tokens[0], tokens[1], int(tokens[2]))
        except Exception:
            logger.log("Failed to process event message", success=logger.FAILED)
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
        by the beamline library. We simply send the ASNC request and return

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

    def read_property(self, property: str) -> Any:
        """
        Read a property value.

        Parameters
        ----------
        property : str
            The property name.

        Returns
        -------
        Any
            The property value converted to Python type.
        """
        ret = self._process_return(
            self._send_receive(f"{CMD_PROPERTY_READ} {property}")
        )
        return self._to_python_value(ret)

    def write_property(self, property: str, value: Any) -> Any:
        """
        Write a property value.

        Parameters
        ----------
        property : str
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
            self._send_receive(f"{CMD_PROPERTY_WRITE} {property} {value}")
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

    def on_event(self, name: str, value: Any, timestamp: int) -> None:
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

    def move_and_wait_end_of_move(
        self,
        motor: str,
        position: float,
        timeout: float,
    ) -> None:
        motor_name = motor.replace(" ", "")

        def wait_motor_ready(timeout: float) -> None:
            current_time = time.perf_counter()
            while True:
                state = str(self.read_property(f"{motor_name}State"))
                if state in {"Ready", "LowLim", "HighLim"}:
                    return
                if time.perf_counter() > current_time + timeout:
                    raise TimeoutError(
                        f"Timeout waiting for {motor} to become Ready (state={state})"
                    )
                time.sleep(0.1)

        def do_move(target: float, timeout: float) -> None:
            self.setMotorPosition(motor_name, float(target))
            wait_motor_ready(timeout)

        do_move(position, timeout)

    def __getitem__(self, key: str) -> Any:
        """
        Get a property value.

        Parameters
        ----------
        key : str
            The property name.

        Returns
        -------
        Any
            The property value.
        """
        return self.read_property(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a property value.

        Parameters
        ----------
        key : str
            The property name.
        value : Any
            The value to set.

        Returns
        -------
        None
        """
        self.write_property(key, value)

    def __getattr__(self, name: str):
        """
        Calls a method on the ExporterClient, e.g. client.methodName(params)

        Parameters
        ----------
        name : str
            The method name.

        Returns
        -------
        callable
            A callable that executes the method on the ExporterClient.
        """

        def caller(*args: Any, **kwargs: Any):
            if kwargs:
                raise TypeError(
                    "ExporterClient methods only support positional arguments; "
                    f"got unexpected keyword args: {sorted(kwargs.keys())}"
                )
            return self.execute(name, *args)

        return caller

    def wait_and_check(
        self,
        task_name: str,
        id: int,
        cmd_start: float,
        expected_time: float,
        timeout: float,
    ) -> bool:
        """
        Wait until the exporter `State` == Ready/On or the timeout

        Parameters
        ----------
        task_name : str
            The name of the task being waited on.
        id : int
            The task identifier.
        cmd_start : float
            The command start time (from time.perf_counter()).
        expected_time : float
            The expected time for the task to complete.
        timeout : float
            The maximum time to wait for the task to complete.

        Returns
        -------
        bool
            True if the task completed successfully, False if it timed out.
        """
        real_time = 0.0
        while True:
            time.sleep(0.1)
            real_time = time.perf_counter() - cmd_start
            try:
                state = str(self.read_property("State")).lower()
            except Exception:
                state = ""
            if state == "ready" or state == "on":
                break
            if real_time > timeout:
                logger.log("Server Timeout", success=logger.FAILED)
                break

        actual_time = real_time
        if actual_time >= timeout:
            logger.log(f"{task_name} {id} failed due to Timeout", success=logger.FAILED)
            return False
        elif actual_time >= 3 * expected_time:
            logger.log(
                f"{task_name} {id} was a lot longer than expected: {actual_time:.4f} sec ",
                success=logger.FAILED,
            )
            return True
        else:
            logger.log(
                f"{task_name} {id} passed in {actual_time:.3f} sec ", success=logger.OK
            )
            return True
