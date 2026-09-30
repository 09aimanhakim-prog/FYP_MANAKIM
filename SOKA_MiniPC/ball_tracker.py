"""Front-camera HSV blob/shape ball tracker for the SOKA robot."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import serial


WINDOW = "SOKA Ball Tracker"
TUNING = "Ball Filter Tuning"
CONFIG_PATH = Path(__file__).with_name("ball_colors.json")
COLOR_KEYS = {ord("1"): "yellow", ord("2"): "blue", ord("3"): "red", ord("4"): "orange"}

# Camera/drive geometry. These ratios remain valid only while the camera mounting
# height and tilt remain unchanged. The 30 cm row comes from the supplied ruler
# reference image and should be refined with an actual ball on the floor.
TARGET_X_RATIO = 0.50
CENTER_TOLERANCE_PX = 20
CENTER_RELEASE_PX = 50
ALIGN_SLOW_ZONE_PX = 70
ALIGN_MEDIUM_ZONE_PX = 160
SLOW_STRAFE_COMMAND = 55.0
MEDIUM_STRAFE_COMMAND = 90.0
FAST_STRAFE_COMMAND = 130.0
VISION_ERROR_EMA_ALPHA = 0.15
VISION_DERIVATIVE_EMA_ALPHA = 0.20
VISION_ALIGN_KP = 0.72
VISION_ALIGN_KD = 0.10
STRAFE_ACCEL_PER_FRAME = 9.0
CENTERING_SPEED_SCALE = 0.80
CENTERED_FRAMES_BEFORE_FORWARD = 10
# Rotate-to-ball controller. Positive camera error means the ball is to the
# right; this robot's installed gyro convention requires a positive adjustment.
CAMERA_TO_HEADING_SIGN = 1.0
ALIGN_SETTLE_TIME_S = 0.12
ALIGN_MEASUREMENT_FRAMES = 3
MIN_HEADING_STEP_DEG = 1.2
MAX_HEADING_STEP_DEG = 6.0
OFFCENTER_FRAMES_TO_REALIGN = 5
REFERENCE_30CM_ROW_RATIO = 0.63
CAPTURE_ROW_RATIO = 0.90
FAR_FORWARD_COMMAND = 90.0
NEAR_FORWARD_COMMAND = 60.0


@dataclass
class ColorFilter:
    h_center: int
    h_tolerance: int
    s_min: int
    v_min: int
    min_area: int
    min_circularity: float


@dataclass
class Detection:
    center: tuple[int, int]
    radius: float
    area: float
    circularity: float
    contour: np.ndarray


DEFAULTS = {
    "yellow": ColorFilter(30, 12, 110, 100, 250, 0.55),
    "blue": ColorFilter(110, 15, 100, 70, 250, 0.50),
    "red": ColorFilter(0, 12, 120, 80, 250, 0.50),
    "orange": ColorFilter(15, 10, 120, 100, 250, 0.55),
}


def load_filters() -> dict[str, ColorFilter]:
    if not CONFIG_PATH.exists():
        return {name: ColorFilter(**asdict(value)) for name, value in DEFAULTS.items()}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return {name: ColorFilter(**data[name]) for name in DEFAULTS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Warning: could not load {CONFIG_PATH.name}: {exc}; using defaults")
        return {name: ColorFilter(**asdict(value)) for name, value in DEFAULTS.items()}


def save_filters(filters: dict[str, ColorFilter]) -> None:
    CONFIG_PATH.write_text(
        json.dumps({name: asdict(value) for name, value in filters.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved color settings to {CONFIG_PATH}")


def hue_mask(hsv: np.ndarray, settings: ColorFilter) -> np.ndarray:
    center = settings.h_center % 180
    tolerance = max(0, min(90, settings.h_tolerance))
    low = center - tolerance
    high = center + tolerance
    lower_sv = (settings.s_min, settings.v_min)

    if low < 0:
        first = cv2.inRange(hsv, (0, *lower_sv), (high, 255, 255))
        second = cv2.inRange(hsv, (180 + low, *lower_sv), (179, 255, 255))
        return cv2.bitwise_or(first, second)
    if high > 179:
        first = cv2.inRange(hsv, (low, *lower_sv), (179, 255, 255))
        second = cv2.inRange(hsv, (0, *lower_sv), (high - 180, 255, 255))
        return cv2.bitwise_or(first, second)
    return cv2.inRange(hsv, (low, *lower_sv), (high, 255, 255))


def find_candidates(mask: np.ndarray, settings: ColorFilter) -> list[Detection]:
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings.min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < settings.min_circularity:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 3:
            continue
        fill = area / (math.pi * radius * radius)
        if fill < 0.45:
            continue
        detections.append(Detection((round(x), round(y)), radius, area, circularity, contour))
    return detections


def select_locked_target(
    candidates: list[Detection], previous: Detection | None
) -> Detection | None:
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda item: item.area * item.circularity)

    def score(item: Detection) -> float:
        distance = math.dist(item.center, previous.center)
        return item.area * item.circularity / (1.0 + 0.03 * distance)

    return max(candidates, key=score)


class RobotLink:
    def __init__(self, port: str) -> None:
        self.serial = serial.Serial(port, 115200, timeout=0, write_timeout=0.2)
        self.last_send = 0.0
        print("Serial connected; waiting 3 seconds for ESP32 startup...")
        time.sleep(3.0)
        self.serial.reset_input_buffer()

    def send(self, command: str) -> None:
        self.serial.write(f"{command}\n".encode("ascii"))

    def send_motion(self, x: float, y: float, heading: float | None = None) -> None:
        now = time.monotonic()
        if now - self.last_send >= 0.05:
            if heading is None:
                self.send(f"MOVE,{x:.1f},{y:.1f}")
            else:
                self.send(f"CTRL,{x:.1f},{y:.1f},{heading:.2f}")
            self.last_send = now

    def read_lines(self) -> list[str]:
        lines = []
        while self.serial.in_waiting:
            line = self.serial.readline().decode("utf-8", errors="replace").strip()
            if line and line not in {"OK,MOVE", "OK,CTRL"}:
                lines.append(line)
        return lines

    def stop(self) -> None:
        self.send("STOP")

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.serial.close()


def make_tuning_window(settings: ColorFilter) -> None:
    cv2.namedWindow(TUNING, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNING, 460, 300)
    noop = lambda _value: None
    cv2.createTrackbar("Hue center", TUNING, settings.h_center, 179, noop)
    cv2.createTrackbar("Hue tolerance", TUNING, settings.h_tolerance, 90, noop)
    cv2.createTrackbar("S minimum", TUNING, settings.s_min, 255, noop)
    cv2.createTrackbar("V minimum", TUNING, settings.v_min, 255, noop)
    cv2.createTrackbar("Minimum area", TUNING, settings.min_area, 10000, noop)
    cv2.createTrackbar(
        "Circularity x100", TUNING, round(settings.min_circularity * 100), 100, noop
    )


def settings_from_trackbars() -> ColorFilter:
    return ColorFilter(
        cv2.getTrackbarPos("Hue center", TUNING),
        cv2.getTrackbarPos("Hue tolerance", TUNING),
        cv2.getTrackbarPos("S minimum", TUNING),
        cv2.getTrackbarPos("V minimum", TUNING),
        max(1, cv2.getTrackbarPos("Minimum area", TUNING)),
        cv2.getTrackbarPos("Circularity x100", TUNING) / 100.0,
    )


def apply_settings_to_trackbars(settings: ColorFilter) -> None:
    cv2.setTrackbarPos("Hue center", TUNING, settings.h_center)
    cv2.setTrackbarPos("Hue tolerance", TUNING, settings.h_tolerance)
    cv2.setTrackbarPos("S minimum", TUNING, settings.s_min)
    cv2.setTrackbarPos("V minimum", TUNING, settings.v_min)
    cv2.setTrackbarPos("Minimum area", TUNING, settings.min_area)
    cv2.setTrackbarPos("Circularity x100", TUNING, round(settings.min_circularity * 100))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--serial", help="ESP32 port, for example COM3")
    parser.add_argument("--drive", action="store_true", help="allow robot motion")
    args = parser.parse_args()
    if args.drive and not args.serial:
        parser.error("--drive requires --serial")

    filters = load_filters()
    selected = "orange"
    settings = filters[selected]
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}")
        return 1

    cv2.namedWindow(WINDOW)
    make_tuning_window(settings)
    sampled_hue: int | None = None
    sample_message = ""
    sample_message_until = 0.0
    sample_point: tuple[int, int] | None = None
    current_hsv: np.ndarray | None = None

    def sample_color(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        nonlocal sampled_hue, sample_message, sample_message_until, sample_point
        if event == cv2.EVENT_LBUTTONDOWN and current_hsv is not None:
            y0, y1 = max(0, y - 3), min(current_hsv.shape[0], y + 4)
            x0, x1 = max(0, x - 3), min(current_hsv.shape[1], x + 4)
            sample = current_hsv[y0:y1, x0:x1]
            hue = round(float(np.median(sample[:, :, 0])))
            saturation = round(float(np.median(sample[:, :, 1])))
            value = round(float(np.median(sample[:, :, 2])))
            sampled_hue = hue
            sample_point = (x, y)
            sample_message = f"Sampled {selected.upper()}: H={hue} S={saturation} V={value}"
            sample_message_until = time.monotonic() + 3.0
            print(sample_message)

    cv2.setMouseCallback(WINDOW, sample_color)
    cv2.namedWindow("Ball Mask")
    cv2.setMouseCallback("Ball Mask", sample_color)
    link = RobotLink(args.serial) if args.serial else None
    armed = bool(args.drive)
    target: Detection | None = None
    lost_frames = 0
    filtered_error_px = 0.0
    filtered_error_velocity = 0.0
    previous_filtered_error = 0.0
    previous_frame_time = time.monotonic()
    smoothed_x_command = 0.0
    centered_frames = 0
    center_latched = False
    commanded_heading = 0.0
    align_settle_until = 0.0
    alignment_samples: list[float] = []
    control_mode = "ALIGN"
    offcenter_frames = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame read failed")
                break
            current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if sampled_hue is not None:
                cv2.setTrackbarPos("Hue center", TUNING, sampled_hue)
                sampled_hue = None

            settings = settings_from_trackbars()
            filters[selected] = settings
            mask = hue_mask(current_hsv, settings)
            candidates = find_candidates(mask, settings)
            selected_target = select_locked_target(candidates, target)
            if selected_target is not None:
                target = selected_target
                lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > 8:
                    target = None
                    filtered_error_px = 0.0
                    filtered_error_velocity = 0.0
                    previous_filtered_error = 0.0
                    smoothed_x_command = 0.0
                    centered_frames = 0
                    center_latched = False

            height, width = frame.shape[:2]
            target_x = round(width * TARGET_X_RATIO)
            reference_30cm_y = round(height * REFERENCE_30CM_ROW_RATIO)
            capture_y = round(height * CAPTURE_ROW_RATIO)
            cv2.line(frame, (target_x, 0), (target_x, height), (80, 80, 80), 1)
            zone_top = 88
            for offset, color in (
                (CENTER_TOLERANCE_PX, (0, 255, 0)),
                (ALIGN_SLOW_ZONE_PX, (0, 255, 255)),
                (ALIGN_MEDIUM_ZONE_PX, (0, 140, 255)),
            ):
                cv2.line(frame, (max(0, target_x - offset), zone_top),
                         (max(0, target_x - offset), height), color, 1)
                cv2.line(frame, (min(width - 1, target_x + offset), zone_top),
                         (min(width - 1, target_x + offset), height), color, 1)
            cv2.line(frame, (0, reference_30cm_y), (width, reference_30cm_y),
                     (255, 180, 0), 1)
            cv2.line(frame, (0, capture_y), (width, capture_y), (255, 0, 255), 1)
            cv2.putText(frame, "~30 cm", (5, reference_30cm_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1)
            cv2.putText(frame, "capture line", (5, capture_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)
            cv2.putText(frame, "Zones: GREEN stop | YELLOW slow | ORANGE medium | outside fast",
                        (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (220, 220, 220), 1)
            for candidate in candidates:
                cv2.drawContours(frame, [candidate.contour], -1, (80, 120, 80), 1)

            if target is not None and lost_frames == 0:
                x, y = target.center
                cv2.circle(frame, (x, y), round(target.radius), (0, 255, 0), 2)
                cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                raw_error_px = x - target_x
                filtered_error_px += VISION_ERROR_EMA_ALPHA * (
                    raw_error_px - filtered_error_px
                )
                now = time.monotonic()
                frame_dt = max(0.005, min(0.1, now - previous_frame_time))
                raw_error_velocity = (
                    filtered_error_px - previous_filtered_error
                ) / frame_dt
                filtered_error_velocity += VISION_DERIVATIVE_EMA_ALPHA * (
                    raw_error_velocity - filtered_error_velocity
                )
                previous_filtered_error = filtered_error_px
                previous_frame_time = now
                error_px = filtered_error_px
                error = error_px / (width / 2)
                ball_bottom_ratio = (y + target.radius) / height
                cv2.putText(
                    frame,
                    f"LOCKED  error={error:+.2f} area={target.area:.0f} circ={target.circularity:.2f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                )
                if link and armed:
                    alignment_status = control_mode
                    if control_mode == "ALIGN":
                        centered_frames = 0
                        offcenter_frames = 0
                        if now < align_settle_until:
                            alignment_status = "SETTLING"
                            alignment_samples.clear()
                        else:
                            alignment_status = "MEASURING"
                            alignment_samples.append(float(raw_error_px))
                            if len(alignment_samples) >= ALIGN_MEASUREMENT_FRAMES:
                                measured_error = sum(alignment_samples) / len(alignment_samples)
                                alignment_samples.clear()
                                if abs(measured_error) <= CENTER_TOLERANCE_PX:
                                    control_mode = "APPROACH"
                                    center_latched = True
                                    centered_frames = CENTERED_FRAMES_BEFORE_FORWARD
                                    alignment_status = "APPROACH"
                                else:
                                    normalized_error = min(
                                        1.0, abs(measured_error) / (width / 2)
                                    )
                                    heading_step = MIN_HEADING_STEP_DEG + (
                                        MAX_HEADING_STEP_DEG - MIN_HEADING_STEP_DEG
                                    ) * normalized_error
                                    commanded_heading += (
                                        CAMERA_TO_HEADING_SIGN *
                                        math.copysign(heading_step, measured_error)
                                    )
                                    commanded_heading = (
                                        (commanded_heading + 180.0) % 360.0
                                    ) - 180.0
                                    align_settle_until = now + ALIGN_SETTLE_TIME_S
                                    alignment_status = "NUDGE"
                    else:
                        if abs(error_px) > CENTER_RELEASE_PX:
                            offcenter_frames += 1
                        else:
                            offcenter_frames = 0
                        if offcenter_frames >= OFFCENTER_FRAMES_TO_REALIGN:
                            control_mode = "ALIGN"
                            center_latched = False
                            centered_frames = 0
                            offcenter_frames = 0
                            alignment_samples.clear()
                            align_settle_until = now + ALIGN_SETTLE_TIME_S
                            alignment_status = "REALIGN"

                    centered_stably = control_mode == "APPROACH"
                    if not centered_stably or ball_bottom_ratio >= CAPTURE_ROW_RATIO:
                        y_command = 0.0
                    elif ball_bottom_ratio < REFERENCE_30CM_ROW_RATIO:
                        y_command = FAR_FORWARD_COMMAND
                    else:
                        span = CAPTURE_ROW_RATIO - REFERENCE_30CM_ROW_RATIO
                        remaining = (CAPTURE_ROW_RATIO - ball_bottom_ratio) / span
                        y_command = NEAR_FORWARD_COMMAND + (
                            FAR_FORWARD_COMMAND - NEAR_FORWARD_COMMAND
                        ) * max(0.0, min(1.0, remaining))
                    link.send_motion(0.0, y_command, commanded_heading)
                    cv2.putText(
                        frame,
                        f"{alignment_status} "
                        f"heading={commanded_heading:+.1f} err={error_px:+.0f}px",
                        (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if centered_stably else (0, 255, 255), 2,
                    )
            else:
                cv2.putText(frame, "SEARCHING", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 2)
                if link and armed:
                    link.stop()
                    armed = False
                    smoothed_x_command = 0.0
                    centered_frames = 0
                    center_latched = False
                    commanded_heading = 0.0
                    align_settle_until = 0.0
                    alignment_samples.clear()
                    control_mode = "ALIGN"
                    offcenter_frames = 0
                    print("Target lost: robot STOPPED and tracking disarmed.")

            cv2.putText(
                frame,
                f"Color: {selected.upper()} | Robot: {'ARMED' if armed else 'SAFE'}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255) if armed else (0, 255, 255), 2,
            )
            if sample_point is not None and time.monotonic() < sample_message_until:
                cv2.drawMarker(frame, sample_point, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(frame, sample_message, (10, 82), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 0, 255), 2)
            cv2.putText(frame, "1:Y 2:B 3:R 4:O | click:sample | S:save | SPACE:arm | Q:quit",
                        (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            cv2.imshow(WINDOW, frame)
            cv2.imshow("Ball Mask", mask)

            if link:
                for line in link.read_lines():
                    print(f"ESP32> {line}")

            key = cv2.waitKey(1) & 0xFF
            if key in COLOR_KEYS:
                selected = COLOR_KEYS[key]
                apply_settings_to_trackbars(filters[selected])
                target = None
            elif key in {ord("s"), ord("S")}:
                save_filters(filters)
            elif key == ord(" "):
                if not link:
                    print("Cannot arm: restart with --serial COM3")
                elif armed:
                    link.stop()
                    armed = False
                    smoothed_x_command = 0.0
                    centered_frames = 0
                    center_latched = False
                    commanded_heading = 0.0
                    align_settle_until = 0.0
                    alignment_samples.clear()
                    control_mode = "ALIGN"
                    offcenter_frames = 0
                    print("Robot tracking stopped.")
                elif target is None or lost_frames != 0:
                    print("Cannot arm: obtain a stable green ball lock first.")
                else:
                    armed = True
                    filtered_error_px = float(target.center[0] - target_x)
                    previous_filtered_error = filtered_error_px
                    filtered_error_velocity = 0.0
                    previous_frame_time = time.monotonic()
                    smoothed_x_command = 0.0
                    centered_frames = 0
                    center_latched = False
                    commanded_heading = 0.0
                    align_settle_until = time.monotonic() + ALIGN_SETTLE_TIME_S
                    alignment_samples.clear()
                    control_mode = "ALIGN"
                    offcenter_frames = 0
                    link.send("ZERO")
                    print("Robot tracking ARMED. Press SPACE again for immediate STOP.")
            elif key in {ord("c"), ord("C")} and link:
                armed = False
                link.send("CALIBRATE")
                print("Calibration started. Keep the robot still for about 62 seconds.")
            elif key in {ord("q"), ord("Q"), 27}:
                break
    finally:
        if link:
            link.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
