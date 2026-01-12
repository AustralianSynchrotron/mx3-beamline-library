from __future__ import annotations

from dataclasses import dataclass
import socket
import time
from typing import Any, Iterable


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


@dataclass(frozen=True)
class ExporterAddress:
    host: str
    port: int


class ExporterClient:
    """
    Synchronous Exporter client without threading
    """

    def __init__(self, address: str, port: int, timeout: float = 3.0):
        self.address = ExporterAddress(address, int(port))
        self.timeout = float(timeout)

    # ---- Transport ----
    def _send_receive(self, payload: str) -> str:
        deadline = time.monotonic() + self.timeout
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
        with socket.create_connection(
            (self.address.host, self.address.port), timeout=self.timeout
        ) as sock:
            sock.settimeout(self.timeout)
            sock.sendall((STX + payload + ETX).encode())

    def _handle_event(self, msg: str) -> None:
        # Expected format: EVT:<name>\t<value>\t<timestamp>
        try:
            evtstr = msg[len(EVENT) :]
            tokens = evtstr.split(PARAMETER_SEPARATOR)
            if len(tokens) < 3:
                return
            self.onEvent(tokens[0], tokens[1], int(tokens[2]))
        except Exception:
            # Keep behaviour compatible with upstream: ignore event parsing errors.
            return

    # ---- Protocol helpers ----
    def _process_return(self, ret: str) -> str | None:
        if ret.startswith(RET_ERR):
            raise RuntimeError(ret[len(RET_ERR) :])
        if ret == RET_NULL:
            return None
        if ret.startswith(RET_OK):
            return ret[len(RET_OK) :]
        raise ExporterProtocolError(f"Unexpected reply: {ret!r}")

    def _to_python_value(self, value: Any) -> Any:
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
    def getMethodList(self) -> list[str] | None:
        ret = self._process_return(self._send_receive(CMD_METHOD_LIST))
        if ret is None:
            return None
        parts = ret.split(PARAMETER_SEPARATOR)
        if len(parts) > 1 and parts[-1] == "":
            parts = parts[:-1]
        return parts

    def getPropertyList(self) -> list[str] | None:
        ret = self._process_return(self._send_receive(CMD_PROPERTY_LIST))
        if ret is None:
            return None
        parts = ret.split(PARAMETER_SEPARATOR)
        if len(parts) > 1 and parts[-1] == "":
            parts = parts[:-1]
        return parts

    def getServerObjectName(self) -> str | None:
        return self._process_return(self._send_receive(CMD_NAME))

    def execute(self, method: str, *pars: Any, timeout: float | None = None) -> Any:
        # NOTE: `timeout` is accepted for API compatibility but we currently use the
        # instance timeout for simplicity.
        cmd = f"{CMD_SYNC_CALL} {method} "
        if pars:
            for par in pars:
                if isinstance(par, (list, tuple)):
                    par = self._create_array_parameter(par)
                cmd += f"{par}{PARAMETER_SEPARATOR}"
        ret = self._process_return(self._send_receive(cmd))
        return self._to_python_value(ret)

    def executeAsync(self, method: str, *pars: Any) -> None:
        # Thread-free: we simply send the ASNC request and return.
        # Any completion/result tracking must be done by polling via other properties/methods.
        cmd = f"{CMD_ASNC_CALL} {method} "
        if pars:
            for par in pars:
                if isinstance(par, (list, tuple)):
                    par = self._create_array_parameter(par)
                cmd += f"{par}{PARAMETER_SEPARATOR}"
        self._send_only(cmd)

    def readProperty(self, prop: str) -> Any:
        ret = self._process_return(self._send_receive(f"{CMD_PROPERTY_READ} {prop}"))
        return self._to_python_value(ret)

    def readPropertyAsString(self, prop: str) -> str | None:
        val = self._process_return(self._send_receive(f"{CMD_PROPERTY_READ} {prop}"))
        return None if val is None else str(val)

    def readPropertyAsFloat(self, prop: str) -> float:
        return float(self.readProperty(prop))

    def readPropertyAsInt(self, prop: str) -> int:
        return int(self.readProperty(prop))

    def readPropertyAsBoolean(self, prop: str) -> bool:
        return bool(self.readProperty(prop))

    def readPropertyAsStringArray(self, prop: str) -> list[str] | None:
        ret = self._process_return(self._send_receive(f"{CMD_PROPERTY_READ} {prop}"))
        if ret is None:
            return None
        return self.parse_array(str(ret))

    def writeProperty(self, prop: str, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            value = self._create_array_parameter(value)
        ret = self._process_return(
            self._send_receive(f"{CMD_PROPERTY_WRITE} {prop} {value}")
        )
        return self._to_python_value(ret)

    def parse_array(self, value: str) -> list[str] | None:
        value = str(value)
        if not value.startswith(ARRAY_SEPARATOR):
            return None
        if value == ARRAY_SEPARATOR:
            return []
        value = value.lstrip(ARRAY_SEPARATOR).rstrip(ARRAY_SEPARATOR)
        return value.split(ARRAY_SEPARATOR)

    def _create_array_parameter(self, value: Iterable[Any]) -> str:
        return "".join(f"{item}{PARAMETER_SEPARATOR}" for item in value)

    # ---- Upstream API aliases (to ease transition) ----
    def parseArray(self, value: str) -> list[str] | None:
        return self.parse_array(value)

    def createArrayParameter(self, value: Iterable[Any]) -> str:
        return self._create_array_parameter(value)

    def onEvent(self, name: str, value: Any, timestamp: int) -> None:
        # Override in user code if needed.
        return

    # ---- Convenience methods used by this repo ----
    def retrieveTaskInfo(self, taskId: Any):
        # Kept for backward compatibility with the older GenericClient wrapper.
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
                state = str(self.readProperty(f"{motor_name}State"))
                if state in {"Ready", "LowLim", "HighLim"}:
                    return
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timeout waiting for {motor} to become Ready (state={state})"
                    )
                time.sleep(0.1)

        def do_move(target: float, deadline: float) -> None:
            if useAttr:
                self.writeProperty(f"{motor_name}Position", target)
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
        return self.readProperty(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.writeProperty(key, value)

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


class ClientFactory:
    """Compatibility shim: keep the `ClientFactory.instantiate(...)` call sites."""

    @staticmethod
    def instantiate(*args, **kwargs):
        client_type = kwargs.get("type", "exporter")
        if client_type != "exporter":
            raise ValueError(
                f"Only 'exporter' client is supported; got type={client_type!r}"
            )
        client_args = kwargs.get("args", {})
        return ExporterClient(**client_args)
