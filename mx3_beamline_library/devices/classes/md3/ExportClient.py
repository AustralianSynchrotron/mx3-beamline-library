"""
This code derived from EMBL code is provided AS IS for example purpose
and testing MD Device Server
ARINAX Sep. 2021
"""

try:
    import Queue as queue  # python 2
except ImportError:
    import queue as queue  # python 3
try:
    import thread
except ImportError:
    import _thread as thread

import logging
from threading import Event, RLock

from .Command.embl import ExporterClient as ec
from .Command.embl.StandardClient import PROTOCOL, ProtocolError
from .GenericClient import Attribute, GenericClient


class ExporterClientFactory:
    @staticmethod
    def instantiate(*argc, **kwargs):
        return ExporterClient(*argc, **kwargs)


class ExporterClient(GenericClient):
    global exporter_clients
    exporter_clients = {}

    def __init__(self, address, port):
        super(ExporterClient, self).__init__()
        global exporter_clients
        if not (address, port) in exporter_clients:
            self.client = Exporter(address, port)
            exporter_clients[(address, port)] = self.client
            self.client.start()
        else:
            self.client = exporter_clients[(address, port)]
        # print self.client.get_state()

    def hasCmd(self, cmdName):
        cmdNameLower = cmdName.lower()
        if cmdNameLower in self.cmdCache:
            return True
        methods = self.client.getMethodList()
        methodsName = [cmd.split(" ")[1].split("(")[0].lower() for cmd in methods]
        for method in methodsName:
            if method not in self.cmdCache:
                self.cmdCache.append(method)
        return cmdNameLower in self.cmdCache

    def hasAttribute(self, attrName):
        attrNameLower = attrName.lower()
        if attrNameLower in self.attrCache:
            return True
        properties = self.client.getPropertyList()
        attrNames = (
            attr.split(" ")[1].lower()
            for attr in properties
            if len(attr.split(" ")) > 1
        )
        for attr in attrNames:
            if attr not in self.attrCache:
                self.attrCache.append(attr)
        return attrNameLower in self.attrCache

    def runCmd(self, cmdName, *args, **kwds):
        if cmdName.lower() == "state":
            return self.client.get_state()
        # print("newArgs ", args)
        return self.client.execute(cmdName, *args, **kwds)

    def readAttribute(self, attrName):
        return self.client.readProperty(attrName)

    def writeAttribute(self, attrName, value):
        return self.client.writeProperty(attrName, value)

    def doSubscribe(self, attrName):
        self.client.register(attrName, self.onEventReceived)
        return self.client.callbacks[attrName]

    def doUnsubscribe(self, attrName, eventId):
        del self.client.callbacks[attrName]

    def getAttributesList(self):
        ret = []
        for attr in self.client.getPropertyList():
            splittedParts = attr.split(" ")
            ret.append(
                Attribute(
                    name=splittedParts[1],
                    accessType=splittedParts[2] == "READ_WRITE",
                    rType=splittedParts[0],
                )
            )
        return ret

    def hasEvents(self):
        return True


class Exporter(ec.ExporterClient):
    STATE_EVENT = "State"
    STATUS_EVENT = "Status"
    VALUE_EVENT = "Value"
    POSITION_EVENT = "Position"
    MOTOR_STATES_EVENT = "MotorStates"

    STATE_READY = "Ready"
    STATE_INITIALIZING = "Initializing"
    STATE_STARTING = "Starting"
    STATE_RUNNING = "Running"
    STATE_MOVING = "Moving"
    STATE_CLOSING = "Closing"
    STATE_REMOTE = "Remote"
    STATE_STOPPED = "Stopped"
    STATE_COMMUNICATION_ERROR = "Communication Error"
    STATE_INVALID = "Invalid"
    STATE_OFFLINE = "Offline"
    STATE_ALARM = "Alarm"
    STATE_FAULT = "Fault"
    STATE_UNKNOWN = "Unknown"

    def __init__(self, address, port, timeout=3, retries=1):
        ec.ExporterClient.__init__(
            self, address, port, PROTOCOL.STREAM, timeout, retries
        )

        self.started = False
        self.callbacks = {}
        try:
            self.events_queue = queue.Queue()  # python 2
        except NameError:
            self.events_queue = queue.Queue()  # python 3
        self.events_received = Event()
        self.events_lock = RLock()
        self.events_processing_task = None

    def start(self):
        pass
        # self.started=True
        # self.reconnect()

    def stop(self):
        # self.started=False
        self.disconnect()

    def execute(self, cmdName, *args, **kwargs):
        try:
            arg = args
            if "pars" in kwargs:
                arg += (kwargs["pars"],)
            ret = ec.ExporterClient.execute(self, cmdName, pars=arg)
            return self._to_python_value(ret)
        except ProtocolError:
            raise Exception

    def get_state(self):
        return self.execute("getState")

    def readProperty(self, *args, **kwargs):
        ret = ec.ExporterClient.readProperty(self, *args, **kwargs)
        return self._to_python_value(ret)

    def writeProperty(self, *args, **kwargs):
        ec.ExporterClient.writeProperty(self, *args, **kwargs)

    def reconnect(self):
        return
        # if self.started:
        #    try:
        #        self.disconnect()
        #        self.connect()
        #    except Exception:
        #        time.sleep(1.0)
        #        self.reconnect()

    def onDisconnected(self):
        pass  # self.reconnect()

    def register(self, name, cb):
        if callable(cb):
            self.callbacks.setdefault(name, []).append(cb)
        if not self.events_processing_task:
            # self.events_processing_task = gevent.spawn(self.processEventsFromQueue)
            self.events_processing_task = thread.start_new_thread(
                self.processEventsFromQueue, ()
            )

    def _to_python_value(self, value):
        if value is None:
            return value
        # IK TODO make this with eval
        if "\x1f" in value:
            value = self.parseArray(value)
            try:
                value = list(map(int, value))
            except (TypeError, ValueError):
                try:
                    value = list(map(float, value))
                except (TypeError, ValueError):
                    pass
        else:
            if value == "false":
                value = False
            elif value == "true":
                value = True
            else:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        pass
        return value

    def onEvent(self, name, value, timestamp):
        with self.events_lock:
            self.events_queue.put((name, value, timestamp))
            self.events_received.set()

    def processEventsFromQueue(self):
        while True:
            while not self.events_queue.empty():
                try:
                    name, value, timestamp = self.events_queue.get()
                except Exception:
                    return

                for cb in self.callbacks.get(name, []):
                    try:
                        cb(name, self._to_python_value(value), timestamp)
                    except Exception:
                        logging.exception(
                            "Exception while executing callback for event %s" % name
                        )
                        continue
            with self.events_lock:
                self.events_received.clear()
            self.events_received.wait()
