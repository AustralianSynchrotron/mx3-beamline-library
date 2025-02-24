"""
This code is provided AS IS for example purpose and testing MD Device Server
ARINAX Sep. 2021
"""

import re

# import ipdb;
import struct
import sys
import threading
import time
from threading import Condition

import epics

from . import Logger

# requires PIL to display image attributes -> requires python 3
DISPLAY_IMAGE = False
# TODO change this flag when there is advanced move time analysis
# and the server is in simulation
SIMULATION = False

if DISPLAY_IMAGE:
    from PIL import Image
try:
    from cStringIO import StringIO  # python 2
except ImportError:
    from io import StringIO  # python 3 ATTENTION image display fails with it

TASK_INFO_NAME_IDX = 0
TASK_INFO_START_IDX = 2
TASK_INFO_END_IDX = 3
TASK_INFO_RESULT_IDX = 4
TASK_INFO_EXCEPTION_IDX = 5


class Attribute:  # noqa
    def __init__(self, name, accessType, rType, value=0):
        self.name = name
        self.value = value
        self.writable = accessType
        self.returnType = rType


class GenericClient(object):
    def __init__(self):
        self.registeredAttr = {}
        # dictionary of monitored motors where events will be
        # stored with their timestamps and corresponding motor states
        self.monitoredMotors = {}
        self.monitoredHomingMotors = {}
        self.moveLock = Condition()
        self.taskLock = Condition()

        self.homeStartLock = Condition()
        self.homeStopLock = Condition()
        self.taskErrors = {}
        self.cmdCache = ["State"]
        self.attrCache = []

    def subscribe(self, attrName):
        if not attrName in self.registeredAttr:  # noqa
            self.registeredAttr[attrName] = self.doSubscribe(attrName)

    def unsubscribe(self, attrName):
        if attrName in self.registeredAttr:
            self.doUnsubscribe(attrName, self.registeredAttr[attrName])
            del self.registeredAttr[attrName]

    def unsubscribeAll(self):
        for attr in self.registeredAttr.keys():
            self.unsubscribe(attr)

    def __getattr__(self, name):
        # nameLower = name.lower()
        # if 'EpicsClient.EpicsClient' in str(type(self)):
        #     nameLower = name
        #     print("getattr ", nameLower)
        if self.hasCmd(name):

            def f(*args, **kwds):
                return self.runCmd(name, *args, **kwds)

            return f
        # if self.hasAttribute(name):
        return self.readAttribute(name)

    def __setatttr__(self, name, value):
        if self.hasCmd(name):
            return self.writeAttribute(name, value)
            # raise TypeError('Cannot set the value of a command')
        # if self.hasAttribute(name):
        return self.writeAttribute(name, value)

    def __getitem__(self, key):
        # if self.hasAttribute(key):
        # print(key)
        epics.ca.poll(0.001, 0.1)  # might be replaced by time.sleep(0.1)

        return self.readAttribute(key)

    def __setitem__(self, key, value):

        if self.hasCmd(key):
            return self.writeAttribute(key, value)

        self.writeAttribute(key, value)

    def onEventReceived(self, attrName, value, timestamp):

        if attrName.casefold() == "MotorStates".casefold():
            [x for x in value if x.find("Moving") > 0]

            for m in self.monitoredMotors.keys():
                match = re.match(".*" + m + "=(\w+)", " ".join(value))  # noqa
                if match is not None:
                    if (
                        len(self.monitoredMotors[m]) == 0
                        and match.group(1).casefold() == "Moving".casefold()
                    ):
                        self.monitoredMotors[m].append([timestamp, match.group(1)])
                    if len(self.monitoredMotors[m]) == 1 and (
                        match.group(1).casefold() == "Ready".casefold()
                        or match.group(1).casefold() == "LowLim".casefold()
                        or match.group(1).casefold() == "HighLim".casefold()
                    ):
                        # print(match.group(1))
                        with self.moveLock:
                            self.monitoredMotors[m].append([timestamp, match.group(1)])
                            # print("EVT NO MOTOR MOVING")
                            self.moveLock.notifyAll()
                else:
                    print(m)
                    print("\n".join(value))
            for m in self.monitoredHomingMotors.keys():
                match = re.match(".*" + m + "=(\w+)", " ".join(value))  # noqa

                if match is not None:
                    if match.group(1).casefold() == "Initializing".casefold():
                        self.monitoredHomingMotors[m].append(timestamp)
                    elif match.group(1).casefold() == "Ready".casefold():
                        pass
                else:
                    print(m)
                    print("\n".join(value))
        elif attrName.casefold() == "CurrentPhase".casefold():
            with self.taskLock:
                self.taskLock.notify_all()
        elif attrName.casefold() == "State".casefold():
            with self.moveLock:
                if value.casefold() == "Ready".casefold():
                    self.moveLock.notify_all()
            with self.taskLock:
                self.taskLock.notify_all()
        else:
            with self.moveLock:
                if attrName in self.registeredAttr:
                    self.moveLock.notify_all()
        # print('event %s value %s' % (attrName, value))
        # for val in value:
        #    print str(val)

        # print "evt received "

    def scanPlateRange(self, timeout=20):
        """Try and reach the corners of the plate..
        Parameters:
        useEvents: should we use events to monitor the move.
        """
        self.hasEvents()
        hasPM = self["HeadType"] == "Plate"
        if hasPM:
            self.movePlateToShelf(0, 0, 0, 120)
            self.movePlateToShelf(7, 0, 0, 120)
            self.movePlateToShelf(7, 11, 0, 120)
            self.movePlateToShelf(0, 11, 0, 120)
            self.movePlateToShelf(7, 0, 0, 120)

    def movePlateToShelf(self, row, col, shelf, timeout=20):
        tid = 0
        useEvents = self.hasEvents()
        if useEvents:
            self.taskLock.acquire()
        time.process_time()
        try:
            tid = self.startMovePlateToShelf(row, col, shelf)
            th = monitorThread(tid, useEvents, self.monitorTaskInfo)
            th.start()
        except Exception:
            Logger.log("An error occurred while moving the plate.", Logger.FAILED)
            if useEvents:
                self.taskLock.release()
            return False
        if useEvents:
            self.taskLock.wait(timeout)
            self.taskLock.release()
        else:
            th.join()

    def homeMotor(self, motor, timeout=20):
        """Launch a homing for the motor and wait for it to end.
        Parameters:
        motor: motor to be homed.
        useEvents: should we use events to monitor the move.
        """
        useEvents = self.hasEvents()
        tid = 0
        try:
            maxHomingTime = self.getHomingTimeout(motor) / 1000.0
        except TypeError:
            maxHomingTime = 60
        print("Homing %s" % (motor))
        mName = motor.replace(" ", "")
        if useEvents:
            self.homeStartLock.acquire()
            self.subscribe("MotorStates")
            self.monitoredHomingMotors[mName] = []
            self.taskLock.acquire()
        start = time.process_time()
        try:
            tid = self.startHomingMotor(motor)
            th = monitorThread(tid, useEvents, self.monitorTaskInfo)
            th.start()
        except Exception as e:
            Logger.log(
                "An error occurred while homing the motor {} : {}".format(motor, e),
                Logger.FAILED,
            )
            if useEvents:
                self.homeStartLock.release()
                self.taskLock.release()
            return False
        tInfo = self.retrieveTaskInfo(tid)
        while len(tInfo) < 7:
            tInfo = self.retrieveTaskInfo(tid)
        if self.retrieveTaskInfo(tid)[TASK_INFO_END_IDX] != "null":
            if useEvents:
                self.homeStartLock.release()
                self.taskLock.release()
            Logger.log("The %s motor homing did not start" % motor, Logger.FAILED)
            return False
        if useEvents:
            self.homeStartLock.wait(120000)
        #            self.moveLock.release()
        else:
            while (
                self.getMotorState(motor).casefold() != "Initializing".casefold()
                and time.process_time() - start < maxHomingTime
            ):
                pass
        while (
            len(self.monitoredHomingMotors[mName]) == 0
            and time.process_time() - start < maxHomingTime
        ):
            pass
        if len(self.monitoredHomingMotors[mName]) == 0:
            Logger.log(
                "The %s motor should be initializing, while state is %s."
                % (motor, self.getMotorState(motor)),
                Logger.FAILED,
            )
        else:
            Logger.log("The %s motor homing was triggered." % (motor), Logger.OK)

        if useEvents:
            self.homeStartLock.release()
            self.unsubscribe("MotorStates")
            self.taskLock.wait(maxHomingTime + timeout)
            del self.monitoredHomingMotors[mName]
            self.taskLock.release()
        else:
            th.join()
        if self.getMotorState(motor) != "Ready":
            Logger.log(
                "The %s motor homing failed, state was %s"
                % (motor, self.getMotorState(motor)),
                Logger.FAILED,
            )
        else:
            Logger.log("The %s motor homing succeeded" % motor, Logger.OK)

    def waitAndCheck(self, task_name, id, cmd_start, expected_time, timeout):
        """
        :param task_name:
        :param id:
        :param cmd_start:
        :param expected_time:
        :param timeout:
        """
        act_time = self.waitAppReady(cmd_start, expected_time, timeout)
        if act_time >= timeout:
            Logger.log(
                "%s %d failed due to Timeout" % (task_name, id), success=Logger.FAILED
            )
            return False
        elif act_time >= 3 * expected_time:
            Logger.log(
                "%s %d was a lot longer than expected: %.4f sec "
                % (task_name, id, act_time),
                success=Logger.FAILED,
            )
            return True
        else:
            Logger.log(
                "%s %d passed in %.3f sec " % (task_name, id, act_time),
                success=Logger.OK,
            )
            return True

    def waitReady(self, timeout):
        """
        Waits that the server is ready with a timeout
        :param int timeout: timeout value in ms
        """
        cmd_start = time.perf_counter()
        while True:

            # somehow similar to a time.sleep(0.1)
            epics.ca.poll(0.0001, 0.4)
            real_time = time.perf_counter() - cmd_start

            if str.lower(self["State"]) == "ready" or str.lower(self["State"]) == "on":
                break

            if real_time > timeout:
                Logger.log("Server Timeout", success="DEBUG")
                break

    def waitAppReady(self, cmd_start, expected_time, timeout):
        """
        Waits that the server is ready and return the actual wait time
        :param cmd_start:
        :param expected_time:
        :param timeout:
        """
        real_time = 0.0
        ctpoint = 0
        while True:
            sys.stdout.write(".")
            ctpoint += 1
            if ctpoint > 120:
                ctpoint = 0
                sys.stdout.write("\n")

            # time.sleep(0.1)
            epics.ca.poll(0.0001, 0.4)
            # time.sleep(0.1)
            real_time = time.perf_counter() - cmd_start
            # print(server.State)
            if (
                str.lower(str(self["State"])) == "ready"
                or str.lower(str(self["State"])) == "on"
            ):
                break
            if real_time > timeout:
                Logger.log("Server Timeout", success="DEBUG")
                break
        sys.stdout.write("\n")
        return real_time

    def syncMoveMotor(self, motor, position, timeout=30, use_events=True):
        """
        :param str motor: name of the motor to move
        :param float position: position where to move the motor
        :param int timeout: move timeout in seconds
        :param bool use_events: a boolean indicating if events
        mechanisms are used for motor monitoring
        """

        # Start move
        motorName = motor.replace(" ", "")
        motorPositionAttribute = motorName + "Position"

        if use_events and self.hasEvents():
            self.moveLock.acquire()
            self.subscribe("MotorStates")
            self.monitoredMotors[motorName] = []
        else:
            pollingStartClock = time.process_time()

        try:
            self[motorPositionAttribute] = position
        except Exception as e:
            Logger.log(
                "The %s motor could not move due to an error %s." % (motor, e),
                Logger.FAILED,
            )
            if use_events and self.hasEvents():
                del self.monitoredMotors[motorName]
                self.moveLock.release()
                self.unsubscribe("MotorStates")

        # Wait end of move

        if use_events and self.hasEvents():
            self.moveLock.wait(timeout)
            # read the last motor state thrown in an event
            motorState = self.monitoredMotors[motorName][-1][1]
            del self.monitoredMotors[motorName]
            self.moveLock.release()
            self.unsubscribe("MotorStates")
        else:
            motorStateAttribute = motorName + "State"
            motorState = self[motorStateAttribute]  # synchronous read
            timeoutExpired = False
            # check if the motor is ready and has not reached a limit
            while (
                motorState != "Ready"
                and motorState != "LowLim"
                and motorState != "HighLim"
            ):
                motorState = self[motorStateAttribute]
                time.sleep(0.1)
                timeoutExpired = (time.perf_counter() - pollingStartClock) > timeout
                if timeoutExpired:
                    raise Exception(
                        "Timeout occurred while moving motor %s to position %.4f"
                        % (motor, position)
                    )
        Logger.trace("Final %s state: %s" % (motor, motorState))

    def moveAndWaitEndOfMove(  # noqa
        self,
        motor,
        initialPos,
        position,
        useAttr,
        n,
        useEvents=False,
        goBack=False,
        timeout=20,
        backMove=False,
    ):
        """
        This function moves a motor slightly and checks the time it took.
        :param str motor: motor to be moved.
        :param float initialPos: initial position of the motor, needed in
        case we ask for the motor to go back to initial position
        :param float position: position to reach
        :param bool useAttr: should we use the attribute to move the motor
        or the method setMotorPosition, both should be tested
        :param bool useEvents: should we use events to monitor the move
        :param int n: number of moves to make, if it is >1, a return move
        will be automatically performed
        :param bool goBack: if True the procedure ends with the motor at
        initialPos, otherwise it ends at position
        :param int timeout: timeout of each move in seconds
        :rtype: None
        """
        # :param float expectedMovetime: time expected for the move as per calculation.
        # (it will be the same for the return move)
        # print("motor move and wait ***** ", motor)
        useEvents = self.hasEvents()
        # Remove space in motor's name for Epics client
        if "EpicsClient.EpicsClient" in str(type(self)):
            motor = motor.replace(" ", "")
        # Move to initial position if necessary
        if abs(initialPos - float(self.getMotorPosition(motor))) > 0.01:
            self.moveAndWaitEndOfMove(
                motor,
                float(self.getMotorPosition(motor)),
                initialPos,
                useAttr,
                1,
                useEvents,
                False,
                timeout,
            )
        try:
            expectedMoveTime = max(0.1, float(self.getMotorMoveTime(motor, position)))
            Logger.trace(
                "The %s motor will move from %.4f to %.4f in %.4f s"
                % (motor, initialPos, position, expectedMoveTime)
            )
        except Exception:
            expectedMoveTime = 0.1
            Logger.trace(
                "The %s motor will move from %.4f to %.4f in unknown time "
                % (motor, initialPos, position)
            )

        mName = motor.replace(" ", "")
        # print("position ", self[''.join([mName, "Position"])])
        for i in range(0, n):
            if not backMove:
                pass_log = "Pass %d." % (i + 1)
            else:
                pass_log = "Back move"
            if useEvents:
                self.moveLock.acquire()
                self.subscribe("MotorStates")
                self.monitoredMotors[mName] = []
            else:
                start = time.process_time()
            try:
                if useAttr:
                    self["".join([mName, "Position"])] = position
                else:
                    self.setMotorPosition([motor, str(position)])
            except Exception as e:
                Logger.log(
                    "The %s motor could not move due to an error %s. %s"
                    % (motor, e, pass_log),
                    Logger.FAILED,
                )
                if useEvents:
                    del self.monitoredMotors[mName]
                    self.moveLock.release()
                    self.unsubscribe("MotorStates")
                    return
            # print("motor ", motor)
            # st = self.getMotorState(motor)
            # If the expected move time is long enough, we check if the motor is moving.
            st = self["".join([mName, "State"])]  # synchronous read
            # print(st)
            if expectedMoveTime > 1:
                # move is slow so we have time to check that the motor started in moving
                if (
                    str(st).casefold() != "Moving".casefold()
                    and len(self.monitoredMotors[mName]) < 2
                ):
                    Logger.log(
                        "The %s motor should be moving, while state is %s. %s"
                        % (motor, st, pass_log),
                        Logger.FAILED,
                    )
                else:
                    Logger.log(
                        "The %s motor movement was triggered. %s" % (motor, pass_log),
                        Logger.OK,
                    )

            if useEvents:
                Logger.trace(
                    "Waiting for events, timeout = %.4f + %d sec"
                    % (expectedMoveTime, timeout)
                )
                self.moveLock.wait(expectedMoveTime + timeout)
                if len(self.monitoredMotors[mName]) != 2:
                    # timeout occurred
                    realMoveTime = -1
                else:
                    realMoveTime = (
                        self.monitoredMotors[mName][1][0]
                        - self.monitoredMotors[mName][0][0]
                    ) / 1000.0
                    state = self.monitoredMotors[mName][1][1]
                del self.monitoredMotors[mName]
                self.moveLock.release()
                self.unsubscribe("MotorStates")
            else:
                Logger.trace("Sleeping for %.4f s " % expectedMoveTime)
                time.sleep(expectedMoveTime)
                while st != "Ready" and st != "LowLim" and st != "HighLim":
                    st = self["".join([mName, "State"])]
                realMoveTime = time.process_time() - start
                state = st
            if realMoveTime == -1:
                log_msg = (
                    "Motor %s: timeout occurred while waiting for end of move with events. %s\n"
                    % (motor, pass_log)
                )
                Logger.log(log_msg, Logger.FAILED)
            elif (
                (abs(realMoveTime - expectedMoveTime) - 0.5) / expectedMoveTime > 0.2
            ) and not SIMULATION:
                log_msg = (
                    "Motor %s: %.4f s elapsed for the movement, "
                    "%.4f expected (delta : %.4f), percentage : %.4f. %s\n"
                    % (
                        motor,
                        realMoveTime,
                        expectedMoveTime,
                        (realMoveTime - expectedMoveTime),
                        100 * (realMoveTime - expectedMoveTime) / expectedMoveTime,
                        pass_log,
                    )
                )
                Logger.log(log_msg, Logger.FAILED)
            elif (
                (abs(realMoveTime - expectedMoveTime) - 0.5) / expectedMoveTime > 0.5
            ) and SIMULATION:
                log_msg = (
                    "Motor %s: %.4f s elapsed for the movement, "
                    "%.4f expected (delta : %.4f), percentage : %.4f. %s.\n"
                    % (
                        motor,
                        realMoveTime,
                        expectedMoveTime,
                        (realMoveTime - expectedMoveTime),
                        100 * (realMoveTime - expectedMoveTime) / expectedMoveTime,
                        pass_log,
                    )
                )
                Logger.log(log_msg, Logger.FAILED)
            else:
                log_msg = (
                    "Motor %s: %.4f s elapsed for the movement, "
                    "%.3f expected (delta : %.4f). %s."
                    % (
                        motor,
                        realMoveTime,
                        expectedMoveTime,
                        (realMoveTime - expectedMoveTime),
                        pass_log,
                    )
                )
                Logger.log(log_msg, Logger.OK)
                Logger.log(
                    "Motor %s ending state was %s.\n" % (motor, state), Logger.NA
                )
            if goBack or i < n - 1:
                self.moveAndWaitEndOfMove(
                    motor, position, initialPos, useAttr, 1, useEvents, backMove=True
                )

    def isMotorStateConsistant(self, motor):
        try:
            att_name = "".join([motor.replace(" ", ""), "State"])
            att = self[att_name]
            # att = float(str_val)
        except Exception:
            Logger.log(
                "Error reading {} state, via attribute {} -> {}".format(
                    motor, att_name, att
                ),
                Logger.FAILED,
            )
            return
        try:
            meth = self.getMotorState(motor)
        except Exception:
            Logger.log(
                "Error reading {} state, via method call.".format(motor), Logger.FAILED
            )
            return
        if meth.lower() != att.lower():
            Logger.log(
                f"Inconsistency while reading {motor} state. {att_name} "
                f"attribute reading : \n Method reading : {meth}.",
                Logger.FAILED,
            )
        else:
            Logger.log(
                "Attribute and method state for motor {} are consistant".format(motor),
                Logger.OK,
            )

    def isMotorPositionConsistant(self, motor):
        try:
            att = float(self["".join([motor.replace(" ", ""), "Position"])])
        except Exception:
            Logger.log(
                "Error reading {} state, via attribute {}".format(
                    motor, "".join([motor.replace(" ", ""), "Position"])
                ),
                Logger.FAILED,
            )
            return
        try:
            meth = float(self.getMotorPosition(motor))
        except Exception:
            Logger.log(
                "Error reading {} state, via method call.".format(motor), Logger.FAILED
            )
            return
        if (meth - att) > 0.1:
            Logger.log(
                "Inconsistency while reading {} position. {} attribute reading : {} \n"
                "Method reading : {}.".format(
                    motor, "".join([motor.replace(" ", ""), "Position"]), att, meth
                ),
                Logger.FAILED,
            )
        else:
            Logger.log(
                "Attribute and method position for motor {} are consistent".format(
                    motor
                ),
                Logger.OK,
            )

    def readAllAttributesAndGetWritables(self):
        """try and read all the attributes and return a list of the writable ones"""
        attributes = self.getAttributesList()

        presentAttributes = list(attributes)
        presentRWAttributes = list(attributes)
        offset = 0
        for idx, attr in enumerate(attributes):
            try:
                attrVal = self[attr.name]
                if type(attrVal) is list and len(attrVal) > 1000:
                    # picture so print it
                    if DISPLAY_IMAGE:
                        Image.open(
                            StringIO(struct.pack("b" * len(attrVal), *attrVal))
                        ).show()
                        Logger.log("Reading attribute %s" % (attr.name), Logger.OK)
                else:
                    Logger.log(
                        "Reading attribute %s value = %s" % (attr.name, attrVal),
                        Logger.OK,
                    )
                if attr.writable is False:
                    presentRWAttributes.pop(idx + offset)
                    offset -= 1
            except Exception as e:
                Logger.log(
                    "Failed to read attribute %s error : %s" % (attr.name, e),
                    Logger.FAILED,
                )
                presentAttributes.pop(idx + offset)
                presentRWAttributes.pop(idx + offset)
                offset -= 1
        return presentRWAttributes

    def testPredefinedPositions(
        self,
        device,
        positions,
        useEvents=lambda self: self.hasEvents(),
        goBack=False,
        timeout=30,
    ):
        """
        This function brings a device on multiple predefined positions
        There is an option to bring the device to its initial position
        (if it was not unknown)
        :param str device: name of the device to test
        :param list positions: list of the positions to be tested
        :param useEvents: ...
        :param bool goBack: a boolean to indicate if we want the device
        back at its initial position after test
        :param int timeout: timeout in seconds for each move of the device
        :rtype None:
        """
        initialPos = self["".join([device, "Position"])]
        lastPosition = initialPos
        for pos in positions:
            lastPosition = self.setPredefinedPosition(device, pos, useEvents, timeout)
        if goBack and initialPos != lastPosition and initialPos != "Unknown":
            self["".join([device, "Position"])] = initialPos

    def setPredefinedPosition(self, device, position, useEvents=False, timeout=30):
        """
        This function sends a command to set a device to one of its predefined positions in MD
        :param str device: name of the device
        :param str position: name of a predefined position
        :param bool useEvents: ...
        :param int timeout: timeout in seconds for the move to be done
        :rtype str: position where the device has been brought
        """
        useEvents = self.hasEvents() and useEvents
        Logger.trace("%s will move to %s" % (str(device), str(position)))
        # position attributes have no space inside, despite that device names often have spaces
        attr_pos_name = (device + "Position").replace(" ", "")
        initial_position = self[attr_pos_name]

        if (
            initial_position is None
            or initial_position.casefold() == "Unknown".casefold()
            or initial_position.casefold() != position.casefold()
        ):

            if useEvents:
                self.subscribe("State")
                self.moveLock.acquire()

            try:
                self[attr_pos_name] = position  # starts the move
            except Exception as e:
                if useEvents:
                    self.unsubscribe("State")
                    self.moveLock.release()
                Logger.log(
                    "Could not move {} to position {} : exception reading {} = {}.".format(
                        device, position, attr_pos_name, e
                    ),
                    Logger.FAILED,
                )
                return self[attr_pos_name]
            sTime = time.process_time()

            if useEvents:
                self.moveLock.wait(timeout)
            else:
                while (
                    self[attr_pos_name] is None
                    or self[attr_pos_name].casefold() != position.casefold()
                ):
                    if self[attr_pos_name] is not None:
                        if time.process_time() - sTime > timeout:
                            break
                        else:
                            if useEvents:
                                self.moveLock.wait(timeout)
            if useEvents:
                self.moveLock.release()
                self.unsubscribe("State")

            if not self.isDeviceInPosition(device, position):
                Logger.log(
                    "Could not move {} to exact position {} : Found={}.".format(
                        device, position, self[attr_pos_name]
                    ),
                    Logger.FAILED,
                )
            else:
                Logger.log(
                    "{} device went into position {}.".format(device, position),
                    Logger.OK,
                )
                return position
        else:
            Logger.log("Device {} already into position {}.".format(device, position))

        return self["".join([device, "Position"])]

    def testPhasePositionning(
        self,
        prevPos,
        positions,
        allPositions=None,
        testedPhases=None,
        useEvents=lambda self: self.hasEvents(),
    ):
        if testedPhases is None:
            # # testedPhases = {i:{(i, j):0} for i in positions.keys()
            # for j in positions.keys() if i != j}
            testedPhases = {}
        if allPositions is None:
            allPositions = positions
        current = self["CurrentPhase"]
        if not positions:
            return self.checkTestedAndChangePhase(
                prevPos, current, allPositions[prevPos], testedPhases, useEvents
            )
        if current not in positions.keys():
            for pos in positions.keys():
                res = self.checkTestedAndChangePhase(
                    pos, current, allPositions[pos], testedPhases, useEvents
                )
                if not res:
                    return False
                newDict = dict(positions)
                del newDict[pos]
                if self.testPhasePositionning(
                    current, newDict, allPositions, testedPhases, useEvents
                ):
                    continue
                else:
                    return False
            return self.checkTestedAndChangePhase(
                prevPos, current, allPositions[prevPos], testedPhases, useEvents
            )
        else:
            newDict = dict(positions)
            del newDict[current]
            return self.testPhasePositionning(
                current, newDict, allPositions, testedPhases, useEvents
            )
        return True

    def checkTestedAndChangePhase(
        self,
        position,
        prevPosition,
        expectedPos,
        testedPhases,
        useEvents=lambda self: self.hasEvents(),
        timeout=120,
    ):
        if position is None:
            return True
        if prevPosition not in testedPhases:
            testedPhases[prevPosition] = []
        if position in testedPhases[prevPosition] or position == prevPosition:
            return True
        testedPhases[prevPosition].append(position)
        return self.changePhase(position, expectedPos, useEvents, timeout)

    def changePhase(self, position, expectedPos, useEvents=False, timeout=120):
        """
        This function is used to change the phase (Transfer, Centring, Data Collection,
        Beam Location) of the microdiff
        :param str position: name of the phase to reach
        :param dict expectedPos: a dictionary of devices with their positions
        in the desired phase
        :param bool useEvents:
        :rtype: bool
        """
        useEvents = self.hasEvents()
        # tid = 0
        if useEvents:
            self.subscribe("CurrentPhase")
            self.taskLock.acquire()

        # Start the task for setting MD to phase position
        try:
            # start a task for going to a specific phase
            tid = self.startSetPhase(position)
            if tid is None:
                raise Exception(
                    "Task for moving to phase %s could not be created." % position
                )
            # monitor the task (to register errors)
            th = monitorThread(tid, useEvents, self.monitorTaskInfo)
            th.start()
        except Exception as e:
            Logger.log(
                "An error occurred while setting the phase {} : {}".format(position, e),
                Logger.FAILED,
            )
            # release lock on tasks before returning
            if useEvents:
                self.taskLock.release()
                self.unsubscribe("CurrentPhase")
            # th.join()
            return False

        # Wait for the phase change to end or timeout to occur
        sTime = time.process_time()
        duration = 0
        timeoutMsg = ""
        # while self['CurrentPhase'] != position and not self.taskErrors.has_key(tid):
        while self["CurrentPhase"] != position and tid not in self.taskErrors:
            duration = time.process_time() - sTime
            if duration > timeout:
                timeoutMsg = " (Timeout)"
                break
            else:
                if useEvents:
                    self.taskLock.wait(timeout)
                else:
                    pass
        if useEvents:
            self.taskLock.release()
            self.unsubscribe("CurrentPhase")
        # Stop monitoring thread
        th.join()

        # Verify if the current phase is the desired one
        if self["CurrentPhase"] != position:
            # if self.taskErrors.has_key(tid):
            if tid in self.taskErrors:
                timeoutMsg = ". Exception occurred : {}".format(self.taskErrors[tid])
                del self.taskErrors[tid]
            Logger.log(
                "Failed to go to position {}{}.".format(position, timeoutMsg),
                Logger.FAILED,
            )
            return False
        else:
            Logger.log(
                "Phase changed to position {} in {:.3f} sec".format(position, duration),
                Logger.OK,
            )

        # Verify that all devices are were they are supposed to be in the desired phase
        for key, value in iter(expectedPos.items()):
            # Let 10s to each device to reach its position
            devWaitStartTime = time.process_time()
            while (
                not self.isDeviceInPosition(key, value)
                and time.process_time() - devWaitStartTime < 10
            ):
                pass
            if self.isDeviceInPosition(key, value):
                Logger.log("Device {} is in position {}.".format(key, value), Logger.OK)
            else:
                Logger.log(
                    "Device {} is in position {}. {} expected.".format(
                        key, self["".join([key, "Position"])], value
                    ),
                    Logger.FAILED,
                )

        self.waitAppReady(time.perf_counter(), 60, 120)
        duration = time.process_time() - sTime
        Logger.log(
            "App Ready after Phase changed to position {} in {:.3f} sec".format(
                position, duration
            ),
            Logger.OK,
        )
        return True

    def toggleIO(self, io, useEvents=lambda self: self.hasEvents(), timeout=10):
        state = self[io]
        try:
            self[io] = not state
        except Exception as e:
            Logger.log("An exception occurred for IO {} : {}.".format(io, e), Logger.NA)
            return
        if self[io] == state:
            Logger.log(
                "IO {} cannot be set to {}.".format(io, not state), Logger.FAILED
            )
        else:
            Logger.log("IO {} toggled to {}.".format(io, not state), Logger.OK)

    def monitorTaskInfo(self, taskId, useEvents=lambda self: self.hasEvents()):
        tInfo = self.retrieveTaskInfo(taskId)
        # if 'EpicsClient.EpicsClient' in str(type(self)):
        #     tInfo = self['LastTaskInfo']
        # print("tInfo", tInfo)
        if tInfo is None:
            Logger.log("Task with ID %s has no task info." % str(taskId), Logger.FAILED)
            self.taskErrors[taskId] = "No task info available."
            return
        while len(tInfo) < 7 or tInfo[TASK_INFO_START_IDX] == "null":
            epics.ca.poll(0.001, 0.1)  # time.sleep(0.01)
            tInfo = self.retrieveTaskInfo(taskId)
        with self.homeStartLock:
            if tInfo[TASK_INFO_EXCEPTION_IDX] != "null":
                self.taskErrors[taskId] = tInfo[TASK_INFO_EXCEPTION_IDX]
            self.homeStartLock.notify_all()
        while len(tInfo) < 7 or tInfo[TASK_INFO_END_IDX] == "null":
            # time.sleep(0.01)
            epics.ca.poll(0.001, 0.1)
            tInfo = self.retrieveTaskInfo(taskId)
        with self.taskLock:
            if tInfo[TASK_INFO_EXCEPTION_IDX] != "null":
                self.taskErrors[taskId] = tInfo[TASK_INFO_EXCEPTION_IDX]
            self.taskLock.notify_all()

    def retrieveTaskInfo(self, taskId):
        if "EpicsClient.EpicsClient" in str(type(self)):
            tinfo = self["LastTaskInfo"]
            # print("tInfo", tinfo)
            return tinfo
        else:
            return self.getTaskInfo(taskId)


class monitorThread(threading.Thread):
    def __init__(self, taskid, useEvents, monitoringFct):
        threading.Thread.__init__(self)
        self.taskId = taskid
        self.useEvents = useEvents
        self.runFct = monitoringFct

    def run(self):
        self.runFct(self.taskId, self.useEvents)
