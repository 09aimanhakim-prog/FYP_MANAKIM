"""Interactive 20 Hz serial controller for the SOKA ESP32 firmware."""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass

import serial
from serial.tools import list_ports


BAUD_RATE = 115_200
SEND_RATE_HZ = 20.0


@dataclass
class ControlState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    active: bool = False
    heading_enabled: bool = True


class RobotSerialController:
    def __init__(self, port: str) -> None:
        self.serial = serial.Serial(
            port=port,
            baudrate=BAUD_RATE,
            timeout=0.1,
            write_timeout=0.5,
        )
        self.state = ControlState()
        self.state_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.log_lock = threading.Lock()
        self.telemetry_log = None
        self.telemetry_log_path: Path | None = None
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.sender = threading.Thread(target=self._sender_loop, daemon=True)

    def start(self) -> None:
        # Opening a serial port commonly resets an ESP32. Allow gyro calibration
        # to complete before accepting movement commands.
        print("Connected. Waiting 3 seconds for ESP32 reset and gyro calibration...")
        time.sleep(3.0)
        self.serial.reset_input_buffer()
        self.reader.start()
        self.sender.start()
        self.send_line("STATUS")

    def send_line(self, line: str) -> None:
        payload = f"{line}\n".encode("ascii")
        with self.write_lock:
            self.serial.write(payload)
            self.serial.flush()

    def set_control(self, x: float, y: float, heading: float) -> None:
        with self.state_lock:
            self.state = ControlState(
                x=max(-255.0, min(255.0, x)),
                y=max(-255.0, min(255.0, y)),
                heading=heading,
                active=True,
                heading_enabled=True,
            )

    def set_move(self, x: float, y: float) -> None:
        with self.state_lock:
            self.state = ControlState(
                x=max(-255.0, min(255.0, x)),
                y=max(-255.0, min(255.0, y)),
                active=True,
                heading_enabled=False,
            )

    def stop(self) -> None:
        with self.state_lock:
            self.state.active = False
            self.state.x = 0.0
            self.state.y = 0.0
        self.send_line("STOP")

    def close(self) -> None:
        self.shutdown_event.set()
        try:
            self.stop()
        except (serial.SerialException, OSError):
            pass
        self.reader.join(timeout=0.5)
        self.sender.join(timeout=0.5)
        self.stop_telemetry_log()
        self.serial.close()

    def start_telemetry_log(self) -> Path:
        with self.log_lock:
            if self.telemetry_log is not None:
                return self.telemetry_log_path
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.telemetry_log_path = Path(__file__).with_name(
                f"heading_telemetry_{stamp}.log"
            )
            self.telemetry_log = self.telemetry_log_path.open(
                "w", encoding="utf-8", buffering=1
            )
            self.telemetry_log.write("# PC_MONOTONIC_S,ESP32_STATUS\n")
            return self.telemetry_log_path

    def stop_telemetry_log(self) -> Path | None:
        with self.log_lock:
            path = self.telemetry_log_path
            if self.telemetry_log is not None:
                self.telemetry_log.close()
            self.telemetry_log = None
            self.telemetry_log_path = None
            return path

    def _sender_loop(self) -> None:
        period = 1.0 / SEND_RATE_HZ
        next_send = time.monotonic()
        next_status = time.monotonic()
        while not self.shutdown_event.is_set():
            with self.state_lock:
                state = ControlState(**vars(self.state))
            if state.active:
                try:
                    if state.heading_enabled:
                        command = f"CTRL,{state.x:.1f},{state.y:.1f},{state.heading:.1f}"
                    else:
                        command = f"MOVE,{state.x:.1f},{state.y:.1f}"
                    self.send_line(command)
                except (serial.SerialException, OSError) as exc:
                    print(f"\nSerial write failed: {exc}", file=sys.stderr)
                    self.shutdown_event.set()
                    return
            with self.log_lock:
                logging_enabled = self.telemetry_log is not None
            now = time.monotonic()
            if logging_enabled and now >= next_status:
                try:
                    self.send_line("STATUS")
                except (serial.SerialException, OSError) as exc:
                    print(f"\nStatus logging failed: {exc}", file=sys.stderr)
                    self.shutdown_event.set()
                    return
                next_status = now + 0.1
            next_send += period
            delay = next_send - time.monotonic()
            if delay > 0:
                self.shutdown_event.wait(delay)
            else:
                next_send = time.monotonic()

    def _reader_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                line = self.serial.readline().decode("utf-8", errors="replace").strip()
            except (serial.SerialException, OSError) as exc:
                print(f"\nSerial read failed: {exc}", file=sys.stderr)
                self.shutdown_event.set()
                return
            # CTRL is intentionally transmitted at 20 Hz. Its acknowledgement is
            # useful on the wire but too noisy for an interactive terminal and
            # can overwrite what the operator is typing.
            if line.startswith("STATUS,"):
                with self.log_lock:
                    if self.telemetry_log is not None:
                        self.telemetry_log.write(f"{time.monotonic():.3f},{line}\n")
                        continue
            if line and line not in {"OK,CTRL", "OK,MOVE"}:
                print(f"ESP32> {line}")


def available_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def choose_port(requested: str | None) -> str:
    if requested:
        return requested
    ports = available_ports()
    if len(ports) == 1:
        print(f"Automatically selected {ports[0]}")
        return ports[0]
    if not ports:
        raise RuntimeError("No serial ports found. Connect the ESP32 by USB.")
    raise RuntimeError(
        "Multiple serial ports found. Select one with --port: " + ", ".join(ports)
    )


def print_help() -> None:
    print(
        """
Commands:
  ctrl X Y HEADING   Continuously transmit CTRL at 20 Hz
  move X Y           Continuously transmit MOVE without enabling heading hold
  stop               Stop continuous transmission and send STOP
  zero               Set the current physical orientation to heading 0
  calibrate          Run the ESP32 60-second stationary gyro calibration
  status             Request telemetry
  drib on|off|PWM    Control dribbler (PWM is 0-255)
  kick               Trigger the 50 ms kicker pulse
  heading_off        Disable heading hold and stop drivetrain
  log on|off         Log STATUS at 10 Hz without flooding the prompt
  help               Show this help
  quit               Send STOP and exit

Examples:
  zero
  ctrl 0 0 70
  ctrl 60 80 0
  stop
""".strip()
    )


def interactive_loop(controller: RobotSerialController) -> None:
    print_help()
    while not controller.shutdown_event.is_set():
        try:
            parts = input("soka> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not parts:
            continue
        name = parts[0].lower()
        try:
            if name == "ctrl" and len(parts) == 4:
                controller.set_control(float(parts[1]), float(parts[2]), float(parts[3]))
                print("Continuous control active. Enter 'stop' to stop.")
            elif name == "move" and len(parts) == 3:
                controller.send_line("HEADING_OFF")
                controller.set_move(float(parts[1]), float(parts[2]))
                print("Continuous movement active with heading hold disabled.")
            elif name == "stop" and len(parts) == 1:
                controller.stop()
            elif name == "zero" and len(parts) == 1:
                controller.send_line("ZERO")
            elif name == "calibrate" and len(parts) == 1:
                with controller.state_lock:
                    controller.state.active = False
                controller.send_line("CALIBRATE")
                print("Keep the robot completely still for about 62 seconds.")
            elif name == "status" and len(parts) == 1:
                controller.send_line("STATUS")
            elif name == "kick" and len(parts) == 1:
                controller.send_line("KICK")
            elif name == "heading_off" and len(parts) == 1:
                with controller.state_lock:
                    controller.state.active = False
                controller.send_line("HEADING_OFF")
            elif name == "log" and len(parts) == 2:
                if parts[1].lower() == "on":
                    path = controller.start_telemetry_log()
                    print(f"Telemetry logging to {path.name}")
                elif parts[1].lower() == "off":
                    path = controller.stop_telemetry_log()
                    print(f"Telemetry log closed: {path.name if path else 'none active'}")
                else:
                    raise ValueError("use 'log on' or 'log off'")
            elif name == "drib" and len(parts) == 2:
                argument = parts[1].upper()
                if argument not in {"ON", "OFF"}:
                    pwm = int(argument)
                    if not 0 <= pwm <= 255:
                        raise ValueError("dribbler PWM must be from 0 to 255")
                    argument = str(pwm)
                controller.send_line(f"DRIBBLER,{argument}")
            elif name == "help" and len(parts) == 1:
                print_help()
            elif name in {"quit", "exit"} and len(parts) == 1:
                return
            else:
                print("Unknown or malformed command. Enter 'help'.")
        except ValueError as exc:
            print(f"Invalid value: {exc}")
        except (serial.SerialException, OSError) as exc:
            print(f"Serial communication failed: {exc}", file=sys.stderr)
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="ESP32 serial port, for example COM5")
    parser.add_argument(
        "--list", action="store_true", help="list serial ports and exit"
    )
    args = parser.parse_args()

    if args.list:
        ports = list_ports.comports()
        if not ports:
            print("No serial ports found.")
        for port in ports:
            print(f"{port.device}: {port.description}")
        return 0

    try:
        port = choose_port(args.port)
        controller = RobotSerialController(port)
        controller.start()
        interactive_loop(controller)
        controller.close()
        print("Stopped safely.")
        return 0
    except (RuntimeError, serial.SerialException, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
