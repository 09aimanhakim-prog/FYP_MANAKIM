"""Browser-based camera tracker and manual joystick for the SOKA robot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import threading
import time

import cv2
import numpy as np
import serial
from flask import Flask, Response, jsonify, render_template, request

from ball_tracker import (
    ALIGN_MEASUREMENT_FRAMES,
    ALIGN_SETTLE_TIME_S,
    CAMERA_TO_HEADING_SIGN,
    CAPTURE_ROW_RATIO,
    CENTER_RELEASE_PX,
    CENTER_TOLERANCE_PX,
    FAR_FORWARD_COMMAND,
    MAX_HEADING_STEP_DEG,
    MIN_HEADING_STEP_DEG,
    NEAR_FORWARD_COMMAND,
    OFFCENTER_FRAMES_TO_REALIGN,
    REFERENCE_30CM_ROW_RATIO,
    TARGET_X_RATIO,
    find_candidates,
    hue_mask,
    load_filters,
    save_filters,
    select_locked_target,
)
from goal_detector import (
    find_goal_candidates,
    load_goal_filters,
    save_goal_filters,
    select_locked_goal,
)
from side_obstacle import (
    analyze_zone,
    fit_floor_model,
    foreign_pixel_mask,
    load_floor_model,
    save_floor_model,
)
from cone_course import (
    ConeDetection,
    ObstacleTracker,
    detect_black_cones,
    partial_goal_from_mask,
    plan_course_motion,
)
from odometry import EncoderOdometry
from waypoint import plan_waypoint_motion

# Starting estimate for the C920 at the current capture mode. This can later be
# calibrated precisely using an object placed at a known angle.
CAMERA_HORIZONTAL_FOV_DEG = 70.0
MAX_DIRECT_BEARING_DEG = 30.0
HEADING_REACHED_TOLERANCE_DEG = 2.0
HEADING_MOVE_TIMEOUT_S = 1.5

# Continuous autonomous behavior. Search rotates in place around the heading at
# which the target was lost. Capture uses a short timed push because the front
# camera can no longer see a ball once it enters the dribbler.
LOST_FRAMES_BEFORE_SEARCH = 8
SEARCH_STEP_ANGLE_DEG = 60.0
SEARCH_FULL_ROTATION_DEG = 360.0
SEARCH_HEADING_TOLERANCE_DEG = 3.0
SEARCH_LEG_TIMEOUT_S = 2.0
# Check the nearby view before committing to the full 360-degree sweep.
# Positive heading is the physically verified right-turn direction.
SEARCH_PRE_SCAN_OFFSETS_DEG = (50.0, -50.0, 0.0)
SEARCH_PRE_SCAN_LEG_TIMEOUT_S = 1.8
CAPTURE_VISIBLE_ROW_RATIO = CAPTURE_ROW_RATIO - 0.04
CAPTURE_LOST_FRAMES = 2
CAPTURE_PUSH_COMMAND = 55.0
# At Y=55, the 60 mm wheels and 400 RPM command scale predict roughly 27 cm/s
# of forward chassis speed. The floor-tested capture push is currently 1.50 s;
# tune from observed ball retention because acceleration and slip matter.
CAPTURE_PUSH_ESTIMATED_DISTANCE_CM = 30
CAPTURE_PUSH_TIME_S = 1.50
AUTO_DRIBBLER_PWM = 210
WEB_SETTINGS_PATH = Path(__file__).with_name("web_settings.json")

# Near-ball precision alignment. The rear wheel carries most of X motion, while
# both front wheels contribute the required half-X component from omni kinematics.
NEAR_STRAFE_CENTER_PX = 20.0
NEAR_STRAFE_RELEASE_PX = 35.0
NEAR_STRAFE_FULL_ERROR_PX = 120.0
NEAR_STRAFE_MIN_COMMAND = 30.0
NEAR_STRAFE_MAX_COMMAND = 55.0
NEAR_STRAFE_FORWARD_COMMAND = 25.0
NEAR_CENTERED_FORWARD_COMMAND = 50.0
NEAR_ERROR_EMA_ALPHA = 0.20
NEAR_STRAFE_ACCEL_PER_FRAME = 5.0
CAPTURE_CENTER_TOLERANCE_PX = 35.0

# Web ball-color calibration. A click on the ball collects a small central HSV
# patch from several camera frames. Several clicks can be combined before the
# resulting robust HSV limits are saved and used by the normal detector.
BALL_CALIBRATION_FRAMES_PER_SAMPLE = 12
BALL_CALIBRATION_FRAME_INTERVAL_S = 0.075
BALL_CALIBRATION_PATCH_RADIUS_PX = 10
BALL_CALIBRATION_MIN_VALID_PIXELS = 80

# Possession-preserving goal behavior. After the timed ball capture, the robot
# rolls in a circle by combining slow forward motion with a steadily advancing
# gyro target. It never performs the stationary goal-search spin.
GOAL_LOCK_FRAMES = 3
GOAL_LOST_FRAMES_BEFORE_SEARCH = 7
GOAL_SEARCH_FORWARD_COMMAND = 27.0
GOAL_SEARCH_HEADING_RATE_DEG_S = 22.0
GOAL_ALIGN_FORWARD_COMMAND = 22.0
GOAL_APPROACH_FORWARD_COMMAND = 42.0
GOAL_BACKOFF_REVERSE_COMMAND = 28.0
# Enter backoff only when the detected goal bottom is clearly below the shoot
# line, then leave at a smaller margin. The hysteresis prevents forward/reverse
# chatter when camera noise places the goal on either side of one threshold.
GOAL_TOO_CLOSE_ENTER_MARGIN_RATIO = 0.05
GOAL_TOO_CLOSE_EXIT_MARGIN_RATIO = 0.015
GOAL_CENTER_TOLERANCE_PX = 24.0
GOAL_CENTER_RELEASE_PX = 55.0
GOAL_CENTERED_FRAMES_TO_SHOOT = 6
GOAL_MAX_BEARING_DEG = 24.0
GOAL_BEARING_EMA_ALPHA = 0.18
GOAL_SHOOT_ROW_RATIO_DEFAULT = 0.72
GOAL_SHOOT_SETTLE_TIME_S = 0.25
GOAL_POST_KICK_TIME_S = 0.70

GOAL_CALIBRATION_FRAMES_PER_SAMPLE = 12
GOAL_CALIBRATION_FRAME_INTERVAL_S = 0.075
GOAL_CALIBRATION_MIN_VALID_PIXELS = 200

# Second/top camera mask-tuning defaults. Only the two side rectangles are
# active; the robot body, hoops, cables and unused image regions remain masked.
SIDE_MASK_DEFAULTS = {
    "top": 15.0,
    "bottom": 94.0,
    "left_outer": 0.0,
    "left_inner": 25.0,
    "right_inner": 75.0,
    "right_outer": 100.0,
}
FLOOR_CALIBRATION_FRAMES_PER_SAMPLE = 12
FLOOR_CALIBRATION_FRAME_INTERVAL_S = 0.075
FLOOR_CALIBRATION_PATCH_RADIUS_PX = 12
SIDE_OBSTACLE_CONFIRM_FRAMES = 4
SIDE_OBSTACLE_RELEASE_FRAMES = 7
SIDE_AVOID_STRAFE_COMMAND = 35.0

# Goalkeeper mode uses only lateral translation. The center latch gives the
# camera a quiet zone so a few noisy pixels cannot make the robot chatter.
# The ESP32 translation slew limiter provides the final physical acceleration
# ramp, so this layer only calculates the desired lateral command.
GOALKEEPER_CENTER_TOLERANCE_PX = 18.0
GOALKEEPER_CENTER_RELEASE_PX = 30.0
GOALKEEPER_ERROR_EMA_ALPHA = 0.32
GOALKEEPER_STRAFE_KP = 0.38
GOALKEEPER_MIN_STRAFE_COMMAND = 24.0
GOALKEEPER_MAX_STRAFE_COMMAND = 70.0

# Front-camera cone-course defaults. All are adjustable from the SAFE web page.
COURSE_BLACK_VALUE_MAX_DEFAULT = 70
COURSE_ROI_TOP_RATIO_DEFAULT = 0.38
COURSE_MIN_CONE_AREA_DEFAULT = 180.0
COURSE_FORWARD_COMMAND_DEFAULT = 30.0
COURSE_SIDE_DANGER_WIDTH_RATIO_DEFAULT = 0.34
COURSE_ORBIT_START_RATIO_DEFAULT = 0.60
COURSE_ORBIT_PASS_DISTANCE_M_DEFAULT = 0.55
COURSE_GOAL_SEARCH_PROGRESS_DEFAULT = 0.75
COURSE_CONE_HOLD_TIME_S = 0.28
COURSE_GOAL_MEMORY_TIME_S = 4.0
COURSE_GOAL_STRICT_EMA_ALPHA = 0.24
COURSE_GOAL_PARTIAL_EMA_ALPHA = 0.08
COURSE_GOAL_REACHED_FRAMES = 4
COURSE_POST_CLEAR_FORWARD_S_DEFAULT = 0.80
COURSE_RECENTER_START_ERROR_M = 0.12
COURSE_RECENTER_STOP_ERROR_M = 0.06
COURSE_RECENTER_KP = 55.0
COURSE_RECENTER_MIN_COMMAND = 16.0
COURSE_RECENTER_MAX_COMMAND = 34.0
COURSE_RECENTER_FORWARD_COMMAND = 18.0
# Aim the front camera inward while recovering the saved course centerline.
# Positive center error means the centerline is to the robot's field-right.
COURSE_RECENTER_HEADING_KP_DEG_PER_M = 80.0
COURSE_RECENTER_MAX_HEADING_DEG = 28.0
# Tight orbit: keep the tracked obstacle near the center of the camera while
# moving tangentially around it. A small radial correction maintains clearance.
COURSE_ORBIT_CONE_SIDE_RATIO = 0.50
COURSE_ORBIT_TANGENTIAL_COMMAND = 34.0
COURSE_ORBIT_TARGET_BOTTOM_RATIO = 0.83
COURSE_ORBIT_RADIAL_KP = 105.0
COURSE_ORBIT_MAX_FORWARD_COMMAND = 18.0
COURSE_ORBIT_MAX_REVERSE_COMMAND = 16.0
COURSE_ORBIT_MAX_HEADING_STEP_DEG = 10.0
COURSE_ORBIT_LOST_FRAMES_TO_PASS = 6
COURSE_ORBIT_MIN_TIME_S = 0.80
COURSE_ORBIT_FORCE_DISTANCE_MULTIPLIER = 1.55
COURSE_ORBIT_REACQUIRE_COOLDOWN_S = 1.0
COURSE_RESTORE_HEADING_TOLERANCE_DEG = 3.0
COURSE_RESTORE_HEADING_TIMEOUT_S = 2.0
COURSE_GOAL_SCAN_ANGLE_DEG = 38.0
COURSE_GOAL_SCAN_TOLERANCE_DEG = 3.0
COURSE_GOAL_SCAN_LEG_TIMEOUT_S = 1.8
COURSE_GOAL_TRACK_FORWARD_COMMAND = 24.0
COURSE_GOAL_DROPOUT_HOLD_S = 0.35

# Field map, Mini-PC encoder odometry, and clear-floor showcase waypoint.
# Dimensions are user supplied; relative odometry may drift when wheels slip.
FIELD_WIDTH_M_DEFAULT = 3.0
FIELD_LENGTH_M_DEFAULT = 4.0
ODOMETRY_X_SCALE_DEFAULT = 1.0
ODOMETRY_Y_SCALE_DEFAULT = 1.0

# Single-point showcase navigation. The target is field-relative and heading is
# held at zero so the three-wheel drive translates without changing its pose.
WAYPOINT_MAX_COMMAND_DEFAULT = 45.0
WAYPOINT_MIN_COMMAND = 22.0
WAYPOINT_SLOW_RADIUS_M = 0.65
WAYPOINT_ARRIVAL_RADIUS_M_DEFAULT = 0.15
WAYPOINT_ARRIVAL_CONFIRM_FRAMES = 3


class SokaWebController:
    def __init__(self, camera_index: int, serial_port: str | None,
                 side_camera_index: int | None = None) -> None:
        self.lock = threading.RLock()
        self.serial_lock = threading.Lock()
        self.running = True
        self.mode = "safe"
        self.selected_color = "orange"
        self.filters = load_filters()
        self.selected_goal_color = "blue"
        self.goal_filters = load_goal_filters()
        self.frame_jpeg: bytes | None = None
        self.target = None
        self.lost_frames = 0
        self.vision_status = "SEARCHING"
        self.error_px = 0.0
        self.commanded_heading = 0.0
        self.align_settle_until = 0.0
        self.alignment_samples: list[float] = []
        self.control_mode = "ALIGN"
        self.offcenter_frames = 0
        self.waiting_for_heading = False
        self.heading_move_deadline = 0.0
        self.search_center_heading = 0.0
        self.search_direction = -1.0
        self.search_swept_degrees = 0.0
        self.search_leg_deadline = 0.0
        self.search_prescan_index = 0
        self.capture_until = 0.0
        self.last_ball_bottom = 0.0
        self.last_ball_error = 0.0
        self.dribbler_running = False
        self.auto_dribbler_pwm = AUTO_DRIBBLER_PWM
        self.capture_push_time_s = CAPTURE_PUSH_TIME_S
        # The robot may be SAFE while the operator is tuning either the ball
        # page or Course page. Track the selected web page independently so
        # each video feed draws only its relevant guide lines.
        self.display_view = "auto"
        self.reference_row_ratio = REFERENCE_30CM_ROW_RATIO
        self.capture_row_ratio = CAPTURE_VISIBLE_ROW_RATIO
        self.goal_shoot_row_ratio = GOAL_SHOOT_ROW_RATIO_DEFAULT
        self.auto_kick_enabled = False
        self.side_mask = dict(SIDE_MASK_DEFAULTS)
        self.floor_model = load_floor_model()
        self.side_avoidance_enabled = False
        self.course_black_value_max = COURSE_BLACK_VALUE_MAX_DEFAULT
        self.course_roi_top_ratio = COURSE_ROI_TOP_RATIO_DEFAULT
        self.course_min_cone_area = COURSE_MIN_CONE_AREA_DEFAULT
        self.course_forward_command = COURSE_FORWARD_COMMAND_DEFAULT
        self.course_side_danger_width_ratio = (
            COURSE_SIDE_DANGER_WIDTH_RATIO_DEFAULT
        )
        self.course_orbit_start_ratio = COURSE_ORBIT_START_RATIO_DEFAULT
        self.course_orbit_command = COURSE_ORBIT_TANGENTIAL_COMMAND
        self.course_orbit_pass_distance_m = (
            COURSE_ORBIT_PASS_DISTANCE_M_DEFAULT
        )
        self.course_goal_search_progress = (
            COURSE_GOAL_SEARCH_PROGRESS_DEFAULT
        )
        self.course_post_clear_forward_s = COURSE_POST_CLEAR_FORWARD_S_DEFAULT
        self.field_width_m = FIELD_WIDTH_M_DEFAULT
        self.field_length_m = FIELD_LENGTH_M_DEFAULT
        self.odometry_x_scale = ODOMETRY_X_SCALE_DEFAULT
        self.odometry_y_scale = ODOMETRY_Y_SCALE_DEFAULT
        self.waypoint_max_command = WAYPOINT_MAX_COMMAND_DEFAULT
        self.waypoint_arrival_radius_m = WAYPOINT_ARRIVAL_RADIUS_M_DEFAULT
        self._load_web_settings()
        if self.floor_model is None:
            self.side_avoidance_enabled = False
        self.side_frame_jpeg: bytes | None = None
        self.side_camera_connected = False
        self.side_camera_index = side_camera_index
        self.floor_calibration_session = False
        self.floor_calibration_collecting = False
        self.floor_calibration_center = (0, 0)
        self.floor_calibration_frame_count = 0
        self.floor_calibration_points = 0
        self.floor_calibration_last_frame_time = 0.0
        self.floor_calibration_pixels: list[np.ndarray] = []
        self.floor_calibration_burst_pixels: list[np.ndarray] = []
        self.floor_calibration_message = (
            "Start floor calibration, then click clean floor inside a green zone."
        )
        self.floor_calibration_result = ""
        self.side_left_obstacle = False
        self.side_right_obstacle = False
        self.side_left_confirm = 0
        self.side_right_confirm = 0
        self.side_left_release = 0
        self.side_right_release = 0
        self.side_left_area_ratio = 0.0
        self.side_right_area_ratio = 0.0
        self.side_avoidance_state = "FLOOR NOT CALIBRATED"
        self.ball_calibration_session = False
        self.ball_calibration_collecting = False
        self.ball_calibration_color = self.selected_color
        self.ball_calibration_center = (0, 0)
        self.ball_calibration_frame_count = 0
        self.ball_calibration_points = 0
        self.ball_calibration_last_frame_time = 0.0
        self.ball_calibration_pixels: list[np.ndarray] = []
        self.ball_calibration_burst_pixels: list[np.ndarray] = []
        self.ball_calibration_message = "Select a color, start, then click the ball."
        self.ball_calibration_result = ""
        self.goal_calibration_session = False
        self.goal_calibration_collecting = False
        self.goal_calibration_color = self.selected_goal_color
        self.goal_calibration_box = (0, 0, 0, 0)
        self.goal_calibration_frame_count = 0
        self.goal_calibration_points = 0
        self.goal_calibration_last_frame_time = 0.0
        self.goal_calibration_pixels: list[np.ndarray] = []
        self.goal_calibration_burst_pixels: list[np.ndarray] = []
        self.goal_calibration_message = "Select a goal color and start calibration."
        self.goal_calibration_result = ""
        self.goal_target = None
        self.goal_lost_frames = 0
        self.goal_seen_frames = 0
        self.goal_status = "GOAL NOT SEEN"
        self.goal_error_px = 0.0
        self.goal_confidence = 0.0
        self.goal_bottom_ratio = 0.0
        self.goal_search_last_time = time.monotonic()
        self.goal_bearing_filtered = 0.0
        self.goal_bearing_initialized = False
        self.goal_centered_frames = 0
        self.goal_shoot_settle_until = 0.0
        self.goal_kick_sent = False
        self.goal_post_kick_until = 0.0
        self.near_error_filtered = 0.0
        self.near_filter_initialized = False
        self.near_center_latched = False
        self.near_x_command = 0.0
        self.manual_x = 0.0
        self.manual_y = 0.0
        self.manual_r = 0.0
        self.manual_heading = 0.0
        self.manual_control_time = time.monotonic()
        self.manual_updated = 0.0
        self.goalkeeper_error_filtered = 0.0
        self.goalkeeper_filter_initialized = False
        self.goalkeeper_center_latched = False
        self.goalkeeper_x_command = 0.0
        self.course_cones: list[ConeDetection] = []
        self.course_obstacle_tracker = ObstacleTracker()
        self.course_last_cone_time = 0.0
        self.course_goal_x: float | None = None
        self.course_goal_last_seen = 0.0
        self.course_goal_reached_frames = 0
        self.course_goal_reached_latched = False
        self.course_target_x = 0.0
        self.course_x_command = 0.0
        self.course_y_command = 0.0
        self.course_nearest_cone_ratio = 0.0
        self.course_status = "COURSE SAFE"
        self.course_recovery_until = 0.0
        self.course_clear_forward_until = 0.0
        self.course_center_x_m = 0.0
        self.course_center_error_m = 0.0
        self.course_recenter_active = False
        self.course_phase = "CRUISE"
        self.course_pass_side = 0.0
        self.course_locked_obstacle_id = -1
        self.course_locked_cone_x = 0.0
        self.course_locked_cone_bottom = 0.0
        self.course_orbit_start_x_m = 0.0
        self.course_orbit_start_y_m = 0.0
        self.course_orbit_start_time = 0.0
        self.course_orbit_lost_frames = 0
        self.course_orbit_cooldown_until = 0.0
        self.course_restore_deadline = 0.0
        self.course_start_y_m = 0.0
        self.course_progress_ratio = 0.0
        self.course_near_goal = False
        self.course_goal_scan_direction = -1.0
        self.course_goal_scan_deadline = 0.0
        self.odometry = EncoderOdometry()
        self.waypoint_target_x: float | None = None
        self.waypoint_target_y: float | None = None
        self.waypoint_x_command = 0.0
        self.waypoint_y_command = 0.0
        self.waypoint_distance_m = 0.0
        self.waypoint_arrived_frames = 0
        self.waypoint_status = "CLICK A TARGET WHILE SAFE"
        self.last_command_time = 0.0
        self.last_status_request = 0.0
        self.last_serial_line = ""
        self.actual_heading: float | None = None
        self.imu_ok: int | None = None
        self.calibrating = False
        self.calibration_progress = 0
        self.calibration_requested = False
        self.calibration_complete_until = 0.0
        self.serial = None

        if serial_port:
            self.serial = serial.Serial(
                serial_port, 115200, timeout=0, write_timeout=0.2
            )
            time.sleep(3.0)
            self.serial.reset_input_buffer()

        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")

        self.side_capture = None
        self.side_worker = None
        if side_camera_index is not None:
            side_capture = cv2.VideoCapture(side_camera_index, cv2.CAP_DSHOW)
            if not side_capture.isOpened():
                side_capture = cv2.VideoCapture(side_camera_index)
            if side_capture.isOpened():
                side_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                side_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                side_capture.set(cv2.CAP_PROP_FPS, 15)
                self.side_capture = side_capture
                self.side_camera_connected = True
                self.side_worker = threading.Thread(
                    target=self._run_side_camera, daemon=True
                )
                self.side_worker.start()
            else:
                side_capture.release()
                print(f"Warning: could not open side camera {side_camera_index}")

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _send(self, command: str) -> None:
        if self.serial is None:
            return
        try:
            with self.serial_lock:
                self.serial.write(f"{command}\n".encode("ascii"))
        except serial.SerialException as exc:
            with self.lock:
                self.last_serial_line = f"Serial error: {exc}"
                self.mode = "safe"

    def _send_motion(self, command: str) -> None:
        now = time.monotonic()
        if now - self.last_command_time >= 0.05:
            self._send(self._apply_side_avoidance(command))
            self.last_command_time = now

    def _apply_side_avoidance(self, command: str) -> str:
        with self.lock:
            if not self.side_avoidance_enabled or self.mode != "auto":
                return command
            if not command.startswith("CTRL,"):
                return command
            try:
                _name, x_text, y_text, heading_text = command.split(",", 3)
                x = float(x_text)
                y = float(y_text)
                heading = float(heading_text)
            except (TypeError, ValueError):
                return command

            if not self.side_camera_connected or self.floor_model is None:
                self.side_avoidance_state = "SAFETY CAMERA OFFLINE -> STOP"
                x = 0.0
                y = 0.0
            elif self.side_left_obstacle and self.side_right_obstacle:
                x = 0.0
                y = 0.0
            elif self.side_left_obstacle:
                x = SIDE_AVOID_STRAFE_COMMAND
                y = 0.0
            elif self.side_right_obstacle:
                x = -SIDE_AVOID_STRAFE_COMMAND
                y = 0.0
            return f"CTRL,{x:.1f},{y:.1f},{heading:.2f}"

    def set_mode(self, mode: str) -> None:
        if mode not in {"safe", "manual", "auto", "goalkeeper", "course", "waypoint"}:
            raise ValueError("invalid mode")
        with self.lock:
            if mode == "waypoint":
                if self.waypoint_target_x is None or self.waypoint_target_y is None:
                    raise ValueError("click a waypoint on the field map first")
                if not self.odometry.snapshot().valid:
                    raise ValueError("odometry telemetry is not live")
            self._send("STOP")
            self.dribbler_running = False
            self.mode = mode
            self.manual_x = self.manual_y = self.manual_r = 0.0
            self.manual_heading = 0.0
            self.manual_control_time = time.monotonic()
            self.manual_updated = 0.0
            self._reset_auto()
            if mode != "safe":
                self._clear_ball_calibration(
                    "Calibration cancelled because the robot was armed."
                )
                self._clear_goal_calibration(
                    "Calibration cancelled because the robot was armed."
                )
                self._clear_floor_calibration(
                    "Floor calibration cancelled because the robot was armed."
                )
            if mode == "manual":
                self._send("ZERO")
            elif mode == "auto":
                self._send("ZERO")
                self.align_settle_until = time.monotonic() + ALIGN_SETTLE_TIME_S
            elif mode == "goalkeeper":
                self._send("ZERO")
            elif mode == "course":
                self._send("ZERO")
                # The position at arming defines the desired course centerline.
                # It may be nonzero on the field map; only relative X drift
                # from this starting line matters to cone-course recovery.
                self.course_center_x_m = self.odometry.x_m
                self.course_start_y_m = self.odometry.y_m
                self.course_center_error_m = 0.0
                self.course_recenter_active = False
                self.course_phase = "CRUISE"
                self.course_progress_ratio = 0.0
                self.course_near_goal = False
                self.course_goal_scan_direction = -1.0
                self.course_goal_scan_deadline = 0.0
            elif mode == "waypoint":
                # Do not ZERO here: it would rotate the established field frame.
                self.commanded_heading = 0.0
                self.waypoint_status = "WAYPOINT TRAVELLING"

    def _reset_auto(self) -> None:
        self.commanded_heading = 0.0
        self.align_settle_until = 0.0
        self.alignment_samples.clear()
        self.control_mode = "ALIGN"
        self.offcenter_frames = 0
        self.waiting_for_heading = False
        self.heading_move_deadline = 0.0
        self.search_center_heading = 0.0
        self.search_direction = -1.0
        self.search_swept_degrees = 0.0
        self.search_leg_deadline = 0.0
        self.search_prescan_index = 0
        self.capture_until = 0.0
        self.last_ball_bottom = 0.0
        self.last_ball_error = 0.0
        self.near_error_filtered = 0.0
        self.near_filter_initialized = False
        self.near_center_latched = False
        self.near_x_command = 0.0
        self.goal_target = None
        self.goal_lost_frames = 0
        self.goal_seen_frames = 0
        self.goal_status = "GOAL NOT SEEN"
        self.goal_error_px = 0.0
        self.goal_confidence = 0.0
        self.goal_bottom_ratio = 0.0
        self.goal_search_last_time = time.monotonic()
        self.goal_bearing_filtered = 0.0
        self.goal_bearing_initialized = False
        self.goal_centered_frames = 0
        self.goal_shoot_settle_until = 0.0
        self.goal_kick_sent = False
        self.goal_post_kick_until = 0.0
        self.goalkeeper_error_filtered = 0.0
        self.goalkeeper_filter_initialized = False
        self.goalkeeper_center_latched = False
        self.goalkeeper_x_command = 0.0
        self.course_cones.clear()
        self.course_obstacle_tracker.reset()
        self.course_last_cone_time = 0.0
        self.course_goal_x = None
        self.course_goal_last_seen = 0.0
        self.course_goal_reached_frames = 0
        self.course_goal_reached_latched = False
        self.course_target_x = 0.0
        self.course_x_command = 0.0
        self.course_y_command = 0.0
        self.course_nearest_cone_ratio = 0.0
        self.course_status = "COURSE SAFE"
        self.course_recovery_until = 0.0
        self.course_clear_forward_until = 0.0
        self.course_center_error_m = 0.0
        self.course_recenter_active = False
        self.course_phase = "CRUISE"
        self.course_pass_side = 0.0
        self.course_locked_obstacle_id = -1
        self.course_locked_cone_x = 0.0
        self.course_locked_cone_bottom = 0.0
        self.course_orbit_start_x_m = 0.0
        self.course_orbit_start_y_m = 0.0
        self.course_orbit_start_time = 0.0
        self.course_orbit_lost_frames = 0
        self.course_orbit_cooldown_until = 0.0
        self.course_restore_deadline = 0.0
        self.course_progress_ratio = 0.0
        self.course_near_goal = False
        self.course_goal_scan_direction = -1.0
        self.course_goal_scan_deadline = 0.0
        self.waypoint_x_command = 0.0
        self.waypoint_y_command = 0.0
        self.waypoint_distance_m = 0.0
        self.waypoint_arrived_frames = 0
        self.waypoint_status = (
            "TARGET READY" if self.waypoint_target_x is not None
            else "CLICK A TARGET WHILE SAFE"
        )

    @staticmethod
    def _wrap_heading(degrees: float) -> float:
        return ((degrees + 180.0) % 360.0) - 180.0

    def _load_web_settings(self) -> None:
        try:
            data = json.loads(WEB_SETTINGS_PATH.read_text(encoding="utf-8"))
            reference = float(data["reference_row_ratio"])
            capture = float(data["capture_row_ratio"])
            if 0.30 <= reference <= 0.85 and reference + 0.05 <= capture <= 0.95:
                self.reference_row_ratio = reference
                self.capture_row_ratio = capture
            goal_shoot = float(
                data.get("goal_shoot_row_ratio", GOAL_SHOOT_ROW_RATIO_DEFAULT)
            )
            if 0.05 <= goal_shoot <= 0.99:
                self.goal_shoot_row_ratio = goal_shoot
            self.auto_kick_enabled = bool(data.get("auto_kick_enabled", False))
            dribbler_pwm = int(data.get("auto_dribbler_pwm", AUTO_DRIBBLER_PWM))
            if 100 <= dribbler_pwm <= 255:
                self.auto_dribbler_pwm = dribbler_pwm
            capture_push_time = float(
                data.get("capture_push_time_s", CAPTURE_PUSH_TIME_S)
            )
            if 0.25 <= capture_push_time <= 3.0:
                self.capture_push_time_s = capture_push_time
            self.side_avoidance_enabled = bool(
                data.get("side_avoidance_enabled", False)
            )
            side_mask = data.get("side_mask", {})
            candidate = {
                name: float(side_mask.get(name, default))
                for name, default in SIDE_MASK_DEFAULTS.items()
            }
            if self._valid_side_mask(candidate):
                self.side_mask = candidate
            course = data.get("course", {})
            value_max = int(course.get(
                "black_value_max", COURSE_BLACK_VALUE_MAX_DEFAULT
            ))
            roi_top = float(course.get(
                "roi_top_ratio", COURSE_ROI_TOP_RATIO_DEFAULT
            ))
            min_area = float(course.get(
                "min_cone_area", COURSE_MIN_CONE_AREA_DEFAULT
            ))
            forward = float(course.get(
                "forward_command", COURSE_FORWARD_COMMAND_DEFAULT
            ))
            danger_width = float(course.get(
                "side_danger_width_ratio",
                COURSE_SIDE_DANGER_WIDTH_RATIO_DEFAULT,
            ))
            orbit_start = float(course.get(
                "orbit_start_ratio", COURSE_ORBIT_START_RATIO_DEFAULT
            ))
            orbit_distance = float(course.get(
                "orbit_pass_distance_m",
                COURSE_ORBIT_PASS_DISTANCE_M_DEFAULT,
            ))
            orbit_command = float(course.get(
                "orbit_command", COURSE_ORBIT_TANGENTIAL_COMMAND,
            ))
            goal_progress = float(course.get(
                "goal_search_progress",
                COURSE_GOAL_SEARCH_PROGRESS_DEFAULT,
            ))
            post_clear_forward_s = float(course.get(
                "post_clear_forward_s", COURSE_POST_CLEAR_FORWARD_S_DEFAULT
            ))
            if 25 <= value_max <= 130:
                self.course_black_value_max = value_max
            if 0.20 <= roi_top <= 0.70:
                self.course_roi_top_ratio = roi_top
            if 60.0 <= min_area <= 2500.0:
                self.course_min_cone_area = min_area
            if 20.0 <= forward <= 75.0:
                self.course_forward_command = forward
            if 0.15 <= danger_width <= 0.48:
                self.course_side_danger_width_ratio = danger_width
            if 0.45 <= orbit_start <= 0.85:
                self.course_orbit_start_ratio = orbit_start
            if 0.25 <= orbit_distance <= 1.20:
                self.course_orbit_pass_distance_m = orbit_distance
            if 20.0 <= orbit_command <= 65.0:
                self.course_orbit_command = orbit_command
            if 0.50 <= goal_progress <= 0.95:
                self.course_goal_search_progress = goal_progress
            if 0.10 <= post_clear_forward_s <= 3.0:
                self.course_post_clear_forward_s = post_clear_forward_s
            field_map = data.get("field_map", {})
            field_width = float(field_map.get(
                "width_m", FIELD_WIDTH_M_DEFAULT
            ))
            field_length = float(field_map.get(
                "length_m", FIELD_LENGTH_M_DEFAULT
            ))
            x_scale = float(field_map.get(
                "odometry_x_scale", ODOMETRY_X_SCALE_DEFAULT
            ))
            y_scale = float(field_map.get(
                "odometry_y_scale", ODOMETRY_Y_SCALE_DEFAULT
            ))
            waypoint_speed = float(field_map.get(
                "waypoint_max_command", WAYPOINT_MAX_COMMAND_DEFAULT
            ))
            waypoint_radius = float(field_map.get(
                "waypoint_arrival_radius_m",
                WAYPOINT_ARRIVAL_RADIUS_M_DEFAULT,
            ))
            if 0.5 <= field_width <= 20.0:
                self.field_width_m = field_width
            if 0.5 <= field_length <= 30.0:
                self.field_length_m = field_length
            if 0.25 <= x_scale <= 2.5:
                self.odometry_x_scale = x_scale
            if 0.25 <= y_scale <= 2.5:
                self.odometry_y_scale = y_scale
            if 20.0 <= waypoint_speed <= 70.0:
                self.waypoint_max_command = waypoint_speed
            if 0.05 <= waypoint_radius <= 0.50:
                self.waypoint_arrival_radius_m = waypoint_radius
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def _save_web_settings(self) -> None:
        WEB_SETTINGS_PATH.write_text(
            json.dumps(
                {
                    "reference_row_ratio": self.reference_row_ratio,
                    "capture_row_ratio": self.capture_row_ratio,
                    "goal_shoot_row_ratio": self.goal_shoot_row_ratio,
                    "auto_kick_enabled": self.auto_kick_enabled,
                    "auto_dribbler_pwm": self.auto_dribbler_pwm,
                    "capture_push_time_s": self.capture_push_time_s,
                    "side_mask": self.side_mask,
                    "side_avoidance_enabled": self.side_avoidance_enabled,
                    "course": {
                        "black_value_max": self.course_black_value_max,
                        "roi_top_ratio": self.course_roi_top_ratio,
                        "min_cone_area": self.course_min_cone_area,
                        "forward_command": self.course_forward_command,
                        "side_danger_width_ratio": (
                            self.course_side_danger_width_ratio
                        ),
                        "orbit_start_ratio": self.course_orbit_start_ratio,
                        "orbit_pass_distance_m": (
                            self.course_orbit_pass_distance_m
                        ),
                        "orbit_command": self.course_orbit_command,
                        "goal_search_progress": (
                            self.course_goal_search_progress
                        ),
                        "post_clear_forward_s": (
                            self.course_post_clear_forward_s
                        ),
                    },
                    "field_map": {
                        "width_m": self.field_width_m,
                        "length_m": self.field_length_m,
                        "odometry_x_scale": self.odometry_x_scale,
                        "odometry_y_scale": self.odometry_y_scale,
                        "waypoint_max_command": self.waypoint_max_command,
                        "waypoint_arrival_radius_m": (
                            self.waypoint_arrival_radius_m
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def set_detection_lines(self, reference_percent: float,
                            capture_percent: float) -> None:
        reference = reference_percent / 100.0
        capture = capture_percent / 100.0
        if not (0.30 <= reference <= 0.85):
            raise ValueError("near line must be from 30% to 85%")
        if not (0.40 <= capture <= 0.95):
            raise ValueError("capture line must be from 40% to 95%")
        if capture < reference + 0.05:
            raise ValueError("capture line must be at least 5% below near line")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("detection lines can only be changed in SAFE mode")
            self.reference_row_ratio = reference
            self.capture_row_ratio = capture
            self._save_web_settings()

    def set_goal_settings(self, shoot_percent: float,
                          auto_kick_enabled: bool) -> None:
        shoot_ratio = shoot_percent / 100.0
        if not (0.05 <= shoot_ratio <= 0.99):
            raise ValueError("goal shoot line must be from 5% to 99%")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("goal settings can only be changed in SAFE mode")
            self.goal_shoot_row_ratio = shoot_ratio
            self.auto_kick_enabled = bool(auto_kick_enabled)
            self._save_web_settings()

    def set_auto_settings(self, dribbler_pwm: int,
                          capture_push_time_s: float) -> None:
        if not 100 <= dribbler_pwm <= 255:
            raise ValueError("dribbler PWM must be from 100 to 255")
        if not 0.25 <= capture_push_time_s <= 3.0:
            raise ValueError("capture push time must be from 0.25 to 3.00 seconds")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("autonomous settings can only be changed in SAFE mode")
            self.auto_dribbler_pwm = int(dribbler_pwm)
            self.capture_push_time_s = float(capture_push_time_s)
            self._save_web_settings()

    def set_course_settings(self, black_value_max: int,
                            roi_top_percent: float, min_cone_area: float,
                            forward_command: float,
                            goal_stop_percent: float,
                            side_danger_width_percent: float,
                            orbit_start_percent: float,
                            orbit_pass_distance_m: float,
                            orbit_command: float,
                            goal_search_progress_percent: float,
                            post_clear_forward_s: float) -> None:
        roi_top_ratio = roi_top_percent / 100.0
        goal_stop_ratio = goal_stop_percent / 100.0
        side_danger_width_ratio = side_danger_width_percent / 100.0
        orbit_start_ratio = orbit_start_percent / 100.0
        goal_search_progress = goal_search_progress_percent / 100.0
        if not 25 <= black_value_max <= 130:
            raise ValueError("black cutoff must be from 25 to 130")
        if not 0.20 <= roi_top_ratio <= 0.70:
            raise ValueError("cone ROI top must be from 20% to 70%")
        if not 60.0 <= min_cone_area <= 2500.0:
            raise ValueError("minimum cone area must be from 60 to 2500")
        if not 20.0 <= forward_command <= 75.0:
            raise ValueError("course forward command must be from 20 to 75")
        if not 0.05 <= goal_stop_ratio <= 0.99:
            raise ValueError("course goal-stop line must be from 5% to 99%")
        if not 0.15 <= side_danger_width_ratio <= 0.48:
            raise ValueError("side danger width must be from 15% to 48%")
        if not 0.45 <= orbit_start_ratio <= 0.85:
            raise ValueError("cone orbit start must be from 45% to 85%")
        if not 0.25 <= orbit_pass_distance_m <= 1.20:
            raise ValueError("orbit pass distance must be from 0.25 m to 1.20 m")
        if not 20.0 <= orbit_command <= 65.0:
            raise ValueError("orbit movement command must be from 20 to 65")
        if not 0.50 <= goal_search_progress <= 0.95:
            raise ValueError("goal-search progress must be from 50% to 95%")
        if not 0.10 <= post_clear_forward_s <= 3.0:
            raise ValueError("post-clear forward time must be from 0.10 to 3.00 seconds")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("cone settings can only be changed in SAFE mode")
            self.course_black_value_max = int(black_value_max)
            self.course_roi_top_ratio = roi_top_ratio
            self.course_min_cone_area = float(min_cone_area)
            self.course_forward_command = float(forward_command)
            self.goal_shoot_row_ratio = goal_stop_ratio
            self.course_side_danger_width_ratio = side_danger_width_ratio
            self.course_orbit_start_ratio = orbit_start_ratio
            self.course_orbit_pass_distance_m = float(
                orbit_pass_distance_m
            )
            self.course_orbit_command = float(orbit_command)
            self.course_goal_search_progress = goal_search_progress
            self.course_post_clear_forward_s = float(post_clear_forward_s)
            self._save_web_settings()

    def set_field_map(self, width_m: float, length_m: float,
                      odometry_x_scale: float,
                      odometry_y_scale: float,
                      waypoint_max_command: float,
                      waypoint_arrival_radius_m: float) -> None:
        if not 0.5 <= width_m <= 20.0:
            raise ValueError("field width must be from 0.5 m to 20 m")
        if not 0.5 <= length_m <= 30.0:
            raise ValueError("field length must be from 0.5 m to 30 m")
        if not 0.25 <= odometry_x_scale <= 2.5:
            raise ValueError("odometry X scale must be from 0.25 to 2.5")
        if not 0.25 <= odometry_y_scale <= 2.5:
            raise ValueError("odometry Y scale must be from 0.25 to 2.5")
        if not 20.0 <= waypoint_max_command <= 70.0:
            raise ValueError("waypoint speed must be from 20 to 70")
        if not 0.05 <= waypoint_arrival_radius_m <= 0.50:
            raise ValueError("arrival radius must be from 0.05 m to 0.50 m")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("field map settings can only change in SAFE mode")
            self.field_width_m = float(width_m)
            self.field_length_m = float(length_m)
            self.odometry_x_scale = float(odometry_x_scale)
            self.odometry_y_scale = float(odometry_y_scale)
            self.waypoint_max_command = float(waypoint_max_command)
            self.waypoint_arrival_radius_m = float(
                waypoint_arrival_radius_m
            )
            self._save_web_settings()

    def set_waypoint(self, x_m: float, y_m: float) -> None:
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError("waypoint must contain finite coordinates")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("waypoints can only be changed in SAFE mode")
            half_width = self.field_width_m * 0.5
            if not -half_width <= x_m <= half_width:
                raise ValueError("waypoint X is outside the field")
            if not 0.0 <= y_m <= self.field_length_m:
                raise ValueError("waypoint Y is outside the field")
            self.waypoint_target_x = float(x_m)
            self.waypoint_target_y = float(y_m)
            self.waypoint_x_command = 0.0
            self.waypoint_y_command = 0.0
            self.waypoint_distance_m = math.hypot(
                x_m - self.odometry.x_m, y_m - self.odometry.y_m
            )
            self.waypoint_arrived_frames = 0
            self.waypoint_status = "TARGET READY - ARM WAYPOINT"

    def clear_waypoint(self) -> None:
        with self.lock:
            if self.mode != "safe":
                raise ValueError("stop the robot before clearing the waypoint")
            self.waypoint_target_x = None
            self.waypoint_target_y = None
            self.waypoint_x_command = 0.0
            self.waypoint_y_command = 0.0
            self.waypoint_distance_m = 0.0
            self.waypoint_arrived_frames = 0
            self.waypoint_status = "CLICK A TARGET WHILE SAFE"

    def reset_odometry(self) -> None:
        with self.lock:
            if self.mode != "safe":
                raise ValueError("odometry can only be reset in SAFE mode")
            self.odometry.reset(0.0, 0.0)
            self.commanded_heading = 0.0
            self.manual_heading = 0.0
            self.waypoint_target_x = None
            self.waypoint_target_y = None
            self.waypoint_status = "CLICK A TARGET WHILE SAFE"
        # The field origin assumes the robot is facing straight toward +Y.
        # Reset the MPU's relative heading at the same physical starting pose.
        self._send("ZERO")

    def clear_odometry_trail(self) -> None:
        with self.lock:
            self.odometry.clear_trail()

    @staticmethod
    def _valid_side_mask(values: dict[str, float]) -> bool:
        return (
            0.0 <= values["top"] <= 80.0
            and 20.0 <= values["bottom"] <= 100.0
            and values["bottom"] >= values["top"] + 5.0
            and 0.0 <= values["left_outer"] <= 60.0
            and 5.0 <= values["left_inner"] <= 65.0
            and values["left_inner"] >= values["left_outer"] + 3.0
            and 35.0 <= values["right_inner"] <= 95.0
            and 40.0 <= values["right_outer"] <= 100.0
            and values["right_outer"] >= values["right_inner"] + 3.0
            and values["left_inner"] < values["right_inner"]
        )

    def set_side_mask(self, values: dict[str, float]) -> None:
        candidate = {
            name: float(values[name]) for name in SIDE_MASK_DEFAULTS
        }
        if not self._valid_side_mask(candidate):
            raise ValueError("side mask boundaries overlap or are outside their ranges")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("side camera masks can only be changed in SAFE mode")
            self.side_mask = candidate
            self._save_web_settings()

    def set_side_avoidance(self, enabled: bool) -> None:
        with self.lock:
            if self.mode != "safe":
                raise ValueError("side avoidance can only be changed in SAFE mode")
            if enabled and self.floor_model is None:
                raise ValueError("calibrate and save the floor color first")
            if enabled and not self.side_camera_connected:
                raise ValueError("side camera is not connected")
            self.side_avoidance_enabled = bool(enabled)
            self._save_web_settings()

    def _clear_floor_calibration(self, message: str) -> None:
        self.floor_calibration_session = False
        self.floor_calibration_collecting = False
        self.floor_calibration_frame_count = 0
        self.floor_calibration_points = 0
        self.floor_calibration_pixels.clear()
        self.floor_calibration_burst_pixels.clear()
        self.floor_calibration_message = message

    def begin_floor_calibration(self) -> None:
        self.set_mode("safe")
        with self.lock:
            if not self.side_camera_connected:
                raise ValueError("side camera is not connected")
            self.side_avoidance_enabled = False
            self._save_web_settings()
            self.floor_calibration_session = True
            self.floor_calibration_collecting = False
            self.floor_calibration_frame_count = 0
            self.floor_calibration_points = 0
            self.floor_calibration_pixels.clear()
            self.floor_calibration_burst_pixels.clear()
            self.floor_calibration_result = ""
            self.floor_calibration_message = (
                "Click clean floor inside a green zone and keep still for one second."
            )

    def sample_floor_calibration(self, x: float, y: float) -> None:
        with self.lock:
            if self.mode != "safe":
                raise ValueError("floor calibration requires SAFE mode")
            if not self.floor_calibration_session:
                raise ValueError("press Start floor calibration first")
            if self.floor_calibration_collecting:
                raise ValueError("a floor sample is already being captured")
            if not (math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0):
                raise ValueError("invalid floor sample position")
            self.floor_calibration_center = (round(x), round(y))
            self.floor_calibration_frame_count = 0
            self.floor_calibration_burst_pixels.clear()
            self.floor_calibration_last_frame_time = 0.0
            self.floor_calibration_collecting = True
            self.floor_calibration_message = (
                "CAPTURING FLOOR - keep the camera and floor still..."
            )

    def apply_floor_calibration(self) -> None:
        with self.lock:
            if not self.floor_calibration_session:
                raise ValueError("no floor calibration is active")
            if self.floor_calibration_collecting:
                raise ValueError("wait for the current floor sample to finish")
            if not self.floor_calibration_pixels:
                raise ValueError("capture at least one floor sample first")
            pixels = np.concatenate(self.floor_calibration_pixels, axis=0)
            model, summary = fit_floor_model(pixels)
            save_floor_model(model)
            self.floor_model = model
            self.floor_calibration_session = False
            self.floor_calibration_points = 0
            self.floor_calibration_frame_count = 0
            self.floor_calibration_pixels.clear()
            self.floor_calibration_burst_pixels.clear()
            self.floor_calibration_result = summary
            self.floor_calibration_message = (
                f"Saved floor model: {summary}. Preview is active; movement remains OFF."
            )
            self.side_avoidance_state = "CLEAR - MOVEMENT OFF"

    def cancel_floor_calibration(self) -> None:
        with self.lock:
            self._clear_floor_calibration(
                "Floor calibration cancelled; saved model was not changed."
            )

    def _collect_floor_calibration(self, lab: np.ndarray, now: float) -> None:
        with self.lock:
            if not self.floor_calibration_collecting:
                return
            if (now - self.floor_calibration_last_frame_time <
                    FLOOR_CALIBRATION_FRAME_INTERVAL_S):
                return
            x, y = self.floor_calibration_center
            height, width = lab.shape[:2]
            radius = FLOOR_CALIBRATION_PATCH_RADIUS_PX
            values = self.side_mask
            zone_top = height * values["top"] / 100.0
            zone_bottom = height * values["bottom"] / 100.0
            in_left = (
                width * values["left_outer"] / 100.0 <= x
                <= width * values["left_inner"] / 100.0
            )
            in_right = (
                width * values["right_inner"] / 100.0 <= x
                <= width * values["right_outer"] / 100.0
            )
            if not (zone_top <= y <= zone_bottom and (in_left or in_right)):
                self.floor_calibration_collecting = False
                self.floor_calibration_message = (
                    "Click clean floor inside one of the green active zones."
                )
                return
            if x < radius or x >= width - radius or y < radius or y >= height - radius:
                self.floor_calibration_collecting = False
                self.floor_calibration_message = (
                    "Sample was too close to the image edge. Click clean floor again."
                )
                return
            patch = lab[y - radius:y + radius + 1,
                        x - radius:x + radius + 1]
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disk = (xx * xx + yy * yy) <= radius * radius
            self.floor_calibration_burst_pixels.append(patch[disk].copy())
            self.floor_calibration_frame_count += 1
            self.floor_calibration_last_frame_time = now
            if self.floor_calibration_frame_count >= FLOOR_CALIBRATION_FRAMES_PER_SAMPLE:
                burst = np.concatenate(self.floor_calibration_burst_pixels, axis=0)
                self.floor_calibration_pixels.append(burst)
                self.floor_calibration_points += 1
                self.floor_calibration_collecting = False
                self.floor_calibration_message = (
                    f"Floor sample {self.floor_calibration_points} captured. "
                    "Click another clean floor area, or press Save & apply."
                )

    @staticmethod
    def _hysteresis_obstacle(raw: bool, active: bool, confirm: int,
                             release: int) -> tuple[bool, int, int]:
        if raw:
            confirm = min(SIDE_OBSTACLE_CONFIRM_FRAMES, confirm + 1)
            release = 0
            if confirm >= SIDE_OBSTACLE_CONFIRM_FRAMES:
                active = True
        else:
            confirm = 0
            if active:
                release = min(SIDE_OBSTACLE_RELEASE_FRAMES, release + 1)
                if release >= SIDE_OBSTACLE_RELEASE_FRAMES:
                    active = False
                    release = 0
        return active, confirm, release

    def _update_side_obstacles(self, left_raw: bool, right_raw: bool) -> None:
        with self.lock:
            (self.side_left_obstacle, self.side_left_confirm,
             self.side_left_release) = self._hysteresis_obstacle(
                left_raw, self.side_left_obstacle,
                self.side_left_confirm, self.side_left_release,
            )
            (self.side_right_obstacle, self.side_right_confirm,
             self.side_right_release) = self._hysteresis_obstacle(
                right_raw, self.side_right_obstacle,
                self.side_right_confirm, self.side_right_release,
            )
            if self.floor_model is None:
                self.side_avoidance_state = "FLOOR NOT CALIBRATED"
            elif self.side_left_obstacle and self.side_right_obstacle:
                self.side_avoidance_state = (
                    "BOTH SIDES BLOCKED"
                    if self.side_avoidance_enabled
                    else "BOTH SIDES FOREIGN - MOVEMENT OFF"
                )
            elif self.side_left_obstacle:
                self.side_avoidance_state = (
                    "LEFT BLOCKED -> STRAFE RIGHT"
                    if self.side_avoidance_enabled
                    else "LEFT FOREIGN OBJECT - MOVEMENT OFF"
                )
            elif self.side_right_obstacle:
                self.side_avoidance_state = (
                    "RIGHT BLOCKED -> STRAFE LEFT"
                    if self.side_avoidance_enabled
                    else "RIGHT FOREIGN OBJECT - MOVEMENT OFF"
                )
            else:
                suffix = "ACTIVE" if self.side_avoidance_enabled else "MOVEMENT OFF"
                self.side_avoidance_state = f"CLEAR - {suffix}"

    def _capture_lost_row_ratio(self) -> float:
        return max(self.reference_row_ratio, self.capture_row_ratio - 0.06)

    def _clear_ball_calibration(self, message: str) -> None:
        self.ball_calibration_session = False
        self.ball_calibration_collecting = False
        self.ball_calibration_frame_count = 0
        self.ball_calibration_points = 0
        self.ball_calibration_pixels.clear()
        self.ball_calibration_burst_pixels.clear()
        self.ball_calibration_message = message

    def begin_ball_calibration(self, color_name: str) -> None:
        if color_name not in self.filters:
            raise ValueError("invalid color")
        self.set_mode("safe")
        with self.lock:
            self.selected_color = color_name
            self.target = None
            self.ball_calibration_session = True
            self.ball_calibration_collecting = False
            self.ball_calibration_color = color_name
            self.ball_calibration_frame_count = 0
            self.ball_calibration_points = 0
            self.ball_calibration_pixels.clear()
            self.ball_calibration_burst_pixels.clear()
            self.ball_calibration_result = ""
            self.ball_calibration_message = (
                f"Click the center of the {color_name} ball and keep it still for one second."
            )

    def sample_ball_calibration(self, x: float, y: float) -> None:
        with self.lock:
            if self.mode != "safe":
                raise ValueError("ball calibration requires SAFE mode")
            if not self.ball_calibration_session:
                raise ValueError("press Start new calibration first")
            if self.ball_calibration_collecting:
                raise ValueError("a sample burst is already being captured")
            if not (math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0):
                raise ValueError("invalid sample position")
            self.ball_calibration_center = (round(x), round(y))
            self.ball_calibration_frame_count = 0
            self.ball_calibration_burst_pixels.clear()
            self.ball_calibration_last_frame_time = 0.0
            self.ball_calibration_collecting = True
            self.ball_calibration_message = (
                "CAPTURING - keep the ball and camera still..."
            )

    @staticmethod
    def _fit_hsv_thresholds(
        pixels: np.ndarray, minimum_pixels: int
    ) -> tuple[int, int, int, int, str]:
        # Reject very dark and almost-grey pixels first. A saturation-weighted
        # circular histogram finds the dominant hue and handles red across the
        # OpenCV hue wrap at 0/179.
        valid = pixels[(pixels[:, 1] >= 35) & (pixels[:, 2] >= 30)]
        if len(valid) < minimum_pixels:
            raise ValueError(
                "not enough colored pixels; sample farther inside the colored object"
            )
        hues = valid[:, 0].astype(np.int16)
        weights = valid[:, 1].astype(np.float64) / 255.0
        histogram = np.bincount(hues, weights=weights, minlength=180)
        smoothed = sum(np.roll(histogram, offset) for offset in range(-4, 5))
        peak = int(np.argmax(smoothed))
        distance = np.abs(hues - peak)
        distance = np.minimum(distance, 180 - distance)
        cluster = valid[distance <= 22]
        if len(cluster) < minimum_pixels:
            raise ValueError(
                "the sampled area contains too many different colors"
            )

        angles = cluster[:, 0].astype(np.float64) * (2.0 * math.pi / 180.0)
        cluster_weights = cluster[:, 1].astype(np.float64) / 255.0
        angle = math.atan2(
            float(np.sum(np.sin(angles) * cluster_weights)),
            float(np.sum(np.cos(angles) * cluster_weights)),
        )
        center = int(round((angle % (2.0 * math.pi)) * 180.0 / (2.0 * math.pi))) % 180
        hue_distance = np.abs(cluster[:, 0].astype(np.int16) - center)
        hue_distance = np.minimum(hue_distance, 180 - hue_distance)
        tolerance = int(np.clip(math.ceil(np.percentile(hue_distance, 95)) + 3, 6, 28))
        s_min = int(np.clip(np.percentile(cluster[:, 1], 8) - 20, 40, 220))
        v_min = int(np.clip(np.percentile(cluster[:, 2], 8) - 20, 30, 220))

        summary = f"H={center} +/-{tolerance}, S>={s_min}, V>={v_min}"
        return center, tolerance, s_min, v_min, summary

    @classmethod
    def _fit_ball_filter(cls, pixels: np.ndarray, current) -> tuple[object, str]:
        center, tolerance, s_min, v_min, summary = cls._fit_hsv_thresholds(
            pixels, BALL_CALIBRATION_MIN_VALID_PIXELS
        )
        fitted = type(current)(center, tolerance, s_min, v_min,
                               current.min_area, current.min_circularity)
        return fitted, summary

    def apply_ball_calibration(self) -> None:
        with self.lock:
            if not self.ball_calibration_session:
                raise ValueError("no ball calibration is active")
            if self.ball_calibration_collecting:
                raise ValueError("wait for the current sample to finish")
            if not self.ball_calibration_pixels:
                raise ValueError("capture at least one ball sample first")
            color_name = self.ball_calibration_color
            pixels = np.concatenate(self.ball_calibration_pixels, axis=0)
            fitted, summary = self._fit_ball_filter(
                pixels, self.filters[color_name]
            )
            self.filters[color_name] = fitted
            save_filters(self.filters)
            self.target = None
            self.ball_calibration_session = False
            self.ball_calibration_points = 0
            self.ball_calibration_frame_count = 0
            self.ball_calibration_pixels.clear()
            self.ball_calibration_burst_pixels.clear()
            self.ball_calibration_result = summary
            self.ball_calibration_message = (
                f"Saved {color_name.upper()}: {summary}. The tracker is using it now."
            )

    def cancel_ball_calibration(self) -> None:
        with self.lock:
            self._clear_ball_calibration("Calibration cancelled; saved settings were not changed.")

    def _collect_ball_calibration(self, hsv: np.ndarray, now: float) -> None:
        with self.lock:
            if not self.ball_calibration_collecting:
                return
            if (now - self.ball_calibration_last_frame_time <
                    BALL_CALIBRATION_FRAME_INTERVAL_S):
                return
            x, y = self.ball_calibration_center
            height, width = hsv.shape[:2]
            radius = BALL_CALIBRATION_PATCH_RADIUS_PX
            if x < radius or x >= width - radius or y < radius or y >= height - radius:
                self.ball_calibration_collecting = False
                self.ball_calibration_message = (
                    "Sample was too close to the image edge. Click the ball center again."
                )
                return
            patch = hsv[y - radius:y + radius + 1,
                        x - radius:x + radius + 1]
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            disk = (xx * xx + yy * yy) <= radius * radius
            self.ball_calibration_burst_pixels.append(patch[disk].copy())
            self.ball_calibration_frame_count += 1
            self.ball_calibration_last_frame_time = now
            if self.ball_calibration_frame_count >= BALL_CALIBRATION_FRAMES_PER_SAMPLE:
                burst = np.concatenate(self.ball_calibration_burst_pixels, axis=0)
                self.ball_calibration_pixels.append(burst)
                self.ball_calibration_points += 1
                self.ball_calibration_collecting = False
                self.ball_calibration_message = (
                    f"Sample {self.ball_calibration_points} captured. "
                    "Move the ball and click again, or press Save & apply."
                )

    def _clear_goal_calibration(self, message: str) -> None:
        self.goal_calibration_session = False
        self.goal_calibration_collecting = False
        self.goal_calibration_frame_count = 0
        self.goal_calibration_points = 0
        self.goal_calibration_pixels.clear()
        self.goal_calibration_burst_pixels.clear()
        self.goal_calibration_message = message

    def begin_goal_calibration(self, color_name: str) -> None:
        if color_name not in self.goal_filters:
            raise ValueError("invalid goal color")
        self.set_mode("safe")
        with self.lock:
            self._clear_ball_calibration(
                "Ball calibration cancelled because goal calibration started."
            )
            self.selected_goal_color = color_name
            self.goal_target = None
            self.goal_calibration_session = True
            self.goal_calibration_collecting = False
            self.goal_calibration_color = color_name
            self.goal_calibration_result = ""
            self.goal_calibration_message = (
                f"Drag a box inside the colored {color_name} goal area."
            )

    def sample_goal_calibration(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        values = (x1, y1, x2, y2)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("invalid goal sample box")
        left, right = sorted((round(x1), round(x2)))
        top, bottom = sorted((round(y1), round(y2)))
        if right - left < 15 or bottom - top < 15:
            raise ValueError("drag a larger box inside the colored goal area")
        with self.lock:
            if self.mode != "safe":
                raise ValueError("goal calibration requires SAFE mode")
            if not self.goal_calibration_session:
                raise ValueError("press Start new goal calibration first")
            if self.goal_calibration_collecting:
                raise ValueError("a goal sample is already being captured")
            self.goal_calibration_box = (left, top, right, bottom)
            self.goal_calibration_frame_count = 0
            self.goal_calibration_burst_pixels.clear()
            self.goal_calibration_last_frame_time = 0.0
            self.goal_calibration_collecting = True
            self.goal_calibration_message = (
                "CAPTURING GOAL - keep the camera and goal still..."
            )

    def apply_goal_calibration(self) -> None:
        with self.lock:
            if not self.goal_calibration_session:
                raise ValueError("no goal calibration is active")
            if self.goal_calibration_collecting:
                raise ValueError("wait for the current goal sample to finish")
            if not self.goal_calibration_pixels:
                raise ValueError("capture at least one goal sample first")
            color_name = self.goal_calibration_color
            pixels = np.concatenate(self.goal_calibration_pixels, axis=0)
            center, tolerance, s_min, v_min, summary = self._fit_hsv_thresholds(
                pixels, GOAL_CALIBRATION_MIN_VALID_PIXELS
            )
            current = self.goal_filters[color_name]
            self.goal_filters[color_name] = type(current)(
                center,
                tolerance,
                s_min,
                v_min,
                current.min_area,
                current.min_aspect_ratio,
                current.max_aspect_ratio,
                current.min_rectangularity,
            )
            save_goal_filters(self.goal_filters)
            self.goal_target = None
            self.goal_calibration_session = False
            self.goal_calibration_points = 0
            self.goal_calibration_frame_count = 0
            self.goal_calibration_pixels.clear()
            self.goal_calibration_burst_pixels.clear()
            self.goal_calibration_result = summary
            self.goal_calibration_message = (
                f"Saved {color_name.upper()} GOAL: {summary}. Detection is live."
            )

    def cancel_goal_calibration(self) -> None:
        with self.lock:
            self._clear_goal_calibration(
                "Goal calibration cancelled; saved settings were not changed."
            )

    def _collect_goal_calibration(self, hsv: np.ndarray, now: float) -> None:
        with self.lock:
            if not self.goal_calibration_collecting:
                return
            if (now - self.goal_calibration_last_frame_time <
                    GOAL_CALIBRATION_FRAME_INTERVAL_S):
                return
            left, top, right, bottom = self.goal_calibration_box
            height, width = hsv.shape[:2]
            left, right = max(0, left), min(width, right)
            top, bottom = max(0, top), min(height, bottom)
            if right - left < 15 or bottom - top < 15:
                self.goal_calibration_collecting = False
                self.goal_calibration_message = (
                    "Goal sample box was outside the picture. Drag another box."
                )
                return
            inset_x = max(2, round((right - left) * 0.08))
            inset_y = max(2, round((bottom - top) * 0.08))
            roi = hsv[top + inset_y:bottom - inset_y,
                      left + inset_x:right - inset_x]
            # Subsample large regions so multi-sample calibration stays small.
            step = max(1, round(math.sqrt(roi.shape[0] * roi.shape[1] / 5000.0)))
            self.goal_calibration_burst_pixels.append(
                roi[::step, ::step].reshape(-1, 3).copy()
            )
            self.goal_calibration_frame_count += 1
            self.goal_calibration_last_frame_time = now
            if self.goal_calibration_frame_count >= GOAL_CALIBRATION_FRAMES_PER_SAMPLE:
                burst = np.concatenate(self.goal_calibration_burst_pixels, axis=0)
                self.goal_calibration_pixels.append(burst)
                self.goal_calibration_points += 1
                self.goal_calibration_collecting = False
                self.goal_calibration_message = (
                    f"Goal sample {self.goal_calibration_points} captured. "
                    "Drag another box, or press Save & apply."
                )

    def _set_dribbler(self, enabled: bool) -> None:
        if enabled == self.dribbler_running:
            return
        self._send(
            f"DRIBBLER,{self.auto_dribbler_pwm}" if enabled else "DRIBBLER,OFF"
        )
        self.dribbler_running = enabled

    def set_joystick(self, x: float, y: float, rotation: float) -> None:
        with self.lock:
            if self.mode != "manual":
                return
            self.manual_x = max(-100.0, min(100.0, x))
            self.manual_y = max(-100.0, min(100.0, y))
            self.manual_r = max(-100.0, min(100.0, rotation))
            self.manual_updated = time.monotonic()

    def action(self, name: str) -> None:
        commands = {
            "stop": "STOP",
            "zero": "ZERO",
            "kick": "KICK",
            "dribbler_on": f"DRIBBLER,{self.auto_dribbler_pwm}",
            "dribbler_off": "DRIBBLER,OFF",
            "calibrate": "CALIBRATE",
            "status": "STATUS",
        }
        if name not in commands:
            raise ValueError("invalid action")
        if name in {"stop", "calibrate"}:
            with self.lock:
                self.mode = "safe"
                self.manual_x = self.manual_y = 0.0
                self.manual_r = 0.0
                self.dribbler_running = False
        if name == "calibrate":
            with self.lock:
                self.calibrating = True
                self.calibration_progress = 0
                self.calibration_requested = True
                self.calibration_complete_until = 0.0
        if name == "zero":
            with self.lock:
                self.manual_heading = 0.0
                self.manual_control_time = time.monotonic()
        if name == "dribbler_on":
            self.dribbler_running = True
        elif name == "dribbler_off":
            self.dribbler_running = False
        self._send(commands[name])

    def _manual_control(self, now: float) -> None:
        dt = max(0.0, min(0.1, now - self.manual_control_time))
        self.manual_control_time = now
        if now - self.manual_updated > 0.30:
            x = y = rotation = 0.0
        else:
            x, y, rotation = self.manual_x, self.manual_y, self.manual_r
        # Full rotation input changes the requested heading at 90 degrees/second.
        self.manual_heading += (rotation / 100.0) * 90.0 * dt
        self.manual_heading = ((self.manual_heading + 180.0) % 360.0) - 180.0
        self._send_motion(
            f"CTRL,{x:.1f},{y:.1f},{self.manual_heading:.2f}"
        )

    def _stop_goalkeeper_motion(self) -> None:
        was_moving = abs(self.goalkeeper_x_command) > 0.01
        self.goalkeeper_x_command = 0.0
        if was_moving:
            # A center/loss stop is safety-critical enough to bypass the usual
            # 50 ms motion-rate limiter once.
            self._send("CTRL,0.0,0.0,0.00")
            self.last_command_time = time.monotonic()
        else:
            self._send_motion("CTRL,0.0,0.0,0.00")

    def _goalkeeper_control(self, raw_error: float) -> str:
        """Center the detected ball using X only while holding zero heading."""
        self._set_dribbler(False)

        # Raw error owns the stop and direction-change decisions. This prevents
        # an older EMA value from carrying the robot past a newly centered ball.
        if self.goalkeeper_center_latched:
            if abs(raw_error) <= GOALKEEPER_CENTER_RELEASE_PX:
                self._stop_goalkeeper_motion()
                return "GOALKEEPER CENTERED"
            self.goalkeeper_center_latched = False

        if abs(raw_error) <= GOALKEEPER_CENTER_TOLERANCE_PX:
            self.goalkeeper_center_latched = True
            self.goalkeeper_error_filtered = raw_error
            self.goalkeeper_filter_initialized = True
            self._stop_goalkeeper_motion()
            return "GOALKEEPER CENTERED"

        if not self.goalkeeper_filter_initialized:
            self.goalkeeper_error_filtered = raw_error
            self.goalkeeper_filter_initialized = True
        elif raw_error * self.goalkeeper_error_filtered < 0.0:
            self.goalkeeper_error_filtered = raw_error
        else:
            self.goalkeeper_error_filtered += GOALKEEPER_ERROR_EMA_ALPHA * (
                raw_error - self.goalkeeper_error_filtered
            )

        error = self.goalkeeper_error_filtered
        magnitude = max(
            GOALKEEPER_MIN_STRAFE_COMMAND,
            min(GOALKEEPER_MAX_STRAFE_COMMAND,
                abs(error) * GOALKEEPER_STRAFE_KP),
        )
        self.goalkeeper_x_command = math.copysign(magnitude, error)
        self._send_motion(
            f"CTRL,{self.goalkeeper_x_command:.1f},0.0,0.00"
        )
        direction = "RIGHT" if self.goalkeeper_x_command > 0.0 else "LEFT"
        return f"GOALKEEPER STRAFE {direction}"

    def _goalkeeper_without_target(self) -> str:
        # Stop on the first unavailable frame. The firmware watchdog remains a
        # second independent stop if camera processing or serial output fails.
        self.goalkeeper_error_filtered = 0.0
        self.goalkeeper_filter_initialized = False
        self.goalkeeper_center_latched = False
        self._set_dribbler(False)
        self._stop_goalkeeper_motion()
        return "GOALKEEPER WAITING FOR BALL"

    def _stop_course_motion(self) -> None:
        was_moving = (
            abs(self.course_x_command) > 0.01
            or abs(self.course_y_command) > 0.01
        )
        self.course_x_command = 0.0
        self.course_y_command = 0.0
        if was_moving:
            self._send("CTRL,0.0,0.0,0.00")
            self.last_command_time = time.monotonic()
        else:
            self._send_motion("CTRL,0.0,0.0,0.00")

    @staticmethod
    def _field_vector_to_robot(field_x: float, field_y: float,
                               heading_deg: float) -> tuple[float, float]:
        """Rotate a field-frame translation command into the robot frame."""
        theta = math.radians(heading_deg)
        return (
            field_x * math.cos(theta) - field_y * math.sin(theta),
            field_x * math.sin(theta) + field_y * math.cos(theta),
        )

    def _select_orbit_cone(
        self, cones: list[ConeDetection]
    ) -> ConeDetection | None:
        if not cones:
            return None
        if self.course_locked_obstacle_id >= 0:
            for cone in cones:
                if cone.track_id == self.course_locked_obstacle_id:
                    return cone
        if self.course_locked_cone_bottom <= 0.0:
            return max(cones, key=lambda cone: cone.bottom)
        return min(
            cones,
            key=lambda cone: (
                abs(cone.center[0] - self.course_locked_cone_x)
                + 0.65 * abs(cone.bottom - self.course_locked_cone_bottom)
            ),
        )

    def _start_course_orbit(self, cone: ConeDetection, pass_side: float,
                            now: float) -> None:
        odometry = self.odometry.snapshot()
        self.course_phase = "ORBIT"
        self.course_pass_side = 1.0 if pass_side >= 0.0 else -1.0
        self.course_locked_obstacle_id = cone.track_id
        self.course_locked_cone_x = float(cone.center[0])
        self.course_locked_cone_bottom = float(cone.bottom)
        self.course_orbit_start_x_m = odometry.x_m
        self.course_orbit_start_y_m = odometry.y_m
        self.course_orbit_start_time = now
        self.course_orbit_lost_frames = 0
        self.course_recenter_active = False

    def _course_orbit_control(
        self, cones: list[ConeDetection], width: int, height: int, now: float
    ) -> str:
        cone = self._select_orbit_cone(cones)
        odometry = self.odometry.snapshot()
        elapsed = now - self.course_orbit_start_time
        forward_progress = (
            max(0.0, odometry.y_m - self.course_orbit_start_y_m)
            if odometry.valid else 0.0
        )
        cone_bottom_ratio = 0.0

        if cone is not None:
            self.course_orbit_lost_frames = 0
            self.course_locked_cone_x = float(cone.center[0])
            self.course_locked_cone_bottom = float(cone.bottom)
            cone_bottom_ratio = cone.bottom / float(height)
            desired_ratio = COURSE_ORBIT_CONE_SIDE_RATIO
            cone_pixel_error = cone.center[0] - width * desired_ratio
            bearing_step = (
                cone_pixel_error / (width * 0.5)
            ) * (CAMERA_HORIZONTAL_FOV_DEG * 0.5)
            bearing_step = max(
                -COURSE_ORBIT_MAX_HEADING_STEP_DEG,
                min(COURSE_ORBIT_MAX_HEADING_STEP_DEG, bearing_step),
            )
            base_heading = (
                self.actual_heading
                if self.actual_heading is not None else self.commanded_heading
            )
            self.commanded_heading = self._wrap_heading(
                base_heading + CAMERA_TO_HEADING_SIGN * bearing_step
            )
        else:
            self.course_orbit_lost_frames += 1

        passed_by_distance = (
            odometry.valid
            and forward_progress >= self.course_orbit_pass_distance_m
            and elapsed >= COURSE_ORBIT_MIN_TIME_S
        )
        forced_by_distance = (
            odometry.valid
            and forward_progress >= (
                self.course_orbit_pass_distance_m
                * COURSE_ORBIT_FORCE_DISTANCE_MULTIPLIER
            )
        )
        passed_after_loss = (
            self.course_orbit_lost_frames >= COURSE_ORBIT_LOST_FRAMES_TO_PASS
            and elapsed >= COURSE_ORBIT_MIN_TIME_S
        )
        if passed_by_distance or forced_by_distance or passed_after_loss:
            self.course_phase = "RESTORE"
            self.course_orbit_cooldown_until = (
                now + COURSE_ORBIT_REACQUIRE_COOLDOWN_S
            )
            self.course_restore_deadline = (
                now + COURSE_RESTORE_HEADING_TIMEOUT_S
            )
            self.commanded_heading = 0.0
            self.course_x_command = 0.0
            self.course_y_command = 0.0
            self._send_motion("CTRL,0.0,0.0,0.00")
            self.course_status = "OBSTACLE PASSED - LOOKING FORWARD"
            return self.course_status

        robot_x = self.course_pass_side * self.course_orbit_command
        if cone is None:
            robot_y = 0.0
        else:
            radial_error = COURSE_ORBIT_TARGET_BOTTOM_RATIO - cone_bottom_ratio
            robot_y = max(
                -COURSE_ORBIT_MAX_REVERSE_COMMAND,
                min(
                    COURSE_ORBIT_MAX_FORWARD_COMMAND,
                    radial_error * COURSE_ORBIT_RADIAL_KP,
                ),
            )
        self.course_x_command = robot_x
        self.course_y_command = robot_y
        self.course_target_x = self.course_locked_cone_x
        self.course_nearest_cone_ratio = max(
            self.course_nearest_cone_ratio, cone_bottom_ratio
        )
        self._send_motion(
            f"CTRL,{robot_x:.1f},{robot_y:.1f},{self.commanded_heading:.2f}"
        )
        side = "RIGHT" if self.course_pass_side > 0.0 else "LEFT"
        self.course_status = f"COURSE ORBIT {side} - WATCHING OBSTACLE"
        return self.course_status

    def _course_restore_forward(self, now: float) -> str:
        self.commanded_heading = 0.0
        heading_error = (
            self._wrap_heading(-self.actual_heading)
            if self.actual_heading is not None else 999.0
        )
        restored = (
            abs(heading_error) <= COURSE_RESTORE_HEADING_TOLERANCE_DEG
            or now >= self.course_restore_deadline
        )
        self.course_x_command = 0.0
        self.course_y_command = 0.0
        self._send_motion("CTRL,0.0,0.0,0.00")
        if restored:
            self.course_phase = "CRUISE"
            self.course_locked_obstacle_id = -1
            self.course_locked_cone_bottom = 0.0
            self.course_locked_cone_x = 0.0
            self.course_status = "COURSE FACING FORWARD"
        else:
            self.course_status = "COURSE RESTORING FORWARD VIEW"
        return self.course_status

    def _course_goal_seek_control(
        self, observed_goal_x: float | None, width: int, now: float
    ) -> str:
        camera_center = width * 0.5
        if observed_goal_x is not None:
            self.course_phase = "GOAL_TRACK"
            pixel_error = observed_goal_x - camera_center
            bearing = (
                pixel_error / (width * 0.5)
            ) * (CAMERA_HORIZONTAL_FOV_DEG * 0.5)
            bearing = max(-30.0, min(30.0, bearing))
            base_heading = (
                self.actual_heading
                if self.actual_heading is not None else self.commanded_heading
            )
            self.commanded_heading = self._wrap_heading(
                base_heading + CAMERA_TO_HEADING_SIGN * bearing
            )
            forward = (
                COURSE_GOAL_TRACK_FORWARD_COMMAND
                if abs(pixel_error) <= width * 0.16 else 0.0
            )
            self.course_x_command = 0.0
            self.course_y_command = forward
            self._send_motion(
                f"CTRL,0.0,{forward:.1f},{self.commanded_heading:.2f}"
            )
            self.course_status = (
                "COURSE GOAL APPROACH"
                if forward > 0.0 else "COURSE TURNING TO GOAL"
            )
            return self.course_status

        # Do not launch a left/right scan because of one or two dropped goal
        # frames. Hold the last requested bearing briefly, with no translation.
        if (
            self.course_phase == "GOAL_TRACK"
            and now - self.course_goal_last_seen <= COURSE_GOAL_DROPOUT_HOLD_S
        ):
            self.course_x_command = 0.0
            self.course_y_command = 0.0
            self._send_motion(
                f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
            )
            self.course_status = "GOAL TEMPORARILY LOST - HOLDING"
            return self.course_status

        if self.course_phase != "GOAL_SEARCH":
            self.course_phase = "GOAL_SEARCH"
            self.course_goal_scan_direction = -1.0
            self.commanded_heading = -COURSE_GOAL_SCAN_ANGLE_DEG
            self.course_goal_scan_deadline = now + COURSE_GOAL_SCAN_LEG_TIMEOUT_S
        else:
            reached = False
            if self.actual_heading is not None:
                reached = abs(self._wrap_heading(
                    self.commanded_heading - self.actual_heading
                )) <= COURSE_GOAL_SCAN_TOLERANCE_DEG
            if reached or now >= self.course_goal_scan_deadline:
                self.course_goal_scan_direction *= -1.0
                self.commanded_heading = (
                    self.course_goal_scan_direction * COURSE_GOAL_SCAN_ANGLE_DEG
                )
                self.course_goal_scan_deadline = (
                    now + COURSE_GOAL_SCAN_LEG_TIMEOUT_S
                )
        self.course_x_command = 0.0
        self.course_y_command = 0.0
        self._send_motion(
            f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
        )
        direction = "LEFT" if self.commanded_heading < 0.0 else "RIGHT"
        self.course_status = f"NEAR GOAL - LOOKING {direction}"
        return self.course_status

    def _course_control(self, cones: list[ConeDetection], width: int,
                        height: int, observed_goal_x: float | None,
                        goal_bottom_ratio: float, strict_goal: bool,
                        now: float) -> str:
        self._set_dribbler(False)
        camera_center = width * 0.5
        course_odometry = self.odometry.snapshot()
        if course_odometry.valid:
            travelled_y = max(
                0.0, course_odometry.y_m - self.course_start_y_m
            )
            self.course_progress_ratio = max(
                0.0, min(1.5, travelled_y / max(0.5, self.field_length_m))
            )
            self.course_near_goal = (
                self.course_progress_ratio >= self.course_goal_search_progress
            )
        else:
            self.course_progress_ratio = 0.0
            self.course_near_goal = False

        if observed_goal_x is not None:
            alpha = (
                COURSE_GOAL_STRICT_EMA_ALPHA
                if strict_goal else COURSE_GOAL_PARTIAL_EMA_ALPHA
            )
            if self.course_goal_x is None:
                self.course_goal_x = observed_goal_x
            else:
                self.course_goal_x += alpha * (
                    observed_goal_x - self.course_goal_x
                )
            self.course_goal_last_seen = now

        goal_recent = (
            self.course_goal_x is not None
            and now - self.course_goal_last_seen <= COURSE_GOAL_MEMORY_TIME_S
        )
        preferred_x = self.course_goal_x if goal_recent else camera_center

        if strict_goal and goal_bottom_ratio >= self.goal_shoot_row_ratio:
            self.course_goal_reached_frames += 1
        else:
            self.course_goal_reached_frames = 0
        if self.course_goal_reached_frames >= COURSE_GOAL_REACHED_FRAMES:
            self.course_goal_reached_latched = True
            self.course_phase = "GOAL_REACHED"
        if self.course_goal_reached_latched:
            self._stop_course_motion()
            self.course_status = "COURSE GOAL REACHED"
            return self.course_status

        previous_target = (
            self.course_target_x if self.course_target_x > 0.0 else camera_center
        )
        decision = plan_course_motion(
            width,
            height,
            cones,
            preferred_x,
            previous_target,
            self.course_forward_command,
            self.course_orbit_start_ratio,
            self.course_side_danger_width_ratio,
        )

        if self.course_phase == "ORBIT":
            return self._course_orbit_control(cones, width, height, now)
        if self.course_phase == "RESTORE":
            return self._course_restore_forward(now)

        # Start an orbit only for a confirmed object overlapping the projected
        # forward corridor. Edge objects remain visible but are safe to pass.
        corridor_left = width * (0.5 - self.course_side_danger_width_ratio)
        corridor_right = width * (0.5 + self.course_side_danger_width_ratio)
        orbit_candidates = []
        for cone in cones:
            left, _top, obstacle_width, _obstacle_height = cone.bbox
            bottom_ratio = cone.bottom / float(height)
            padding = 14.0 + 55.0 * max(
                0.0, min(1.0, (bottom_ratio - 0.32) / 0.68)
            )
            if left - padding < corridor_right and (
                left + obstacle_width + padding > corridor_left
            ):
                orbit_candidates.append(cone)
        orbit_cone = (
            max(orbit_candidates, key=lambda item: item.bottom)
            if orbit_candidates else None
        )
        orbit_ready = (
            orbit_cone is not None
            and orbit_cone.bottom / float(height) >= self.course_orbit_start_ratio
            and now >= self.course_orbit_cooldown_until
        )
        if orbit_ready:
            if decision.escape_direction != 0.0:
                pass_side = decision.escape_direction
            elif abs(decision.x_command) > 0.01:
                pass_side = 1.0 if decision.x_command > 0.0 else -1.0
            else:
                pass_side = 1.0 if orbit_cone.center[0] < camera_center else -1.0
            self._start_course_orbit(orbit_cone, pass_side, now)
            return self._course_orbit_control(cones, width, height, now)

        if self.course_phase != "RECENTER":
            self.course_pass_side = 0.0

        # Orbit alone handles obstacles; the planner only selects the pass side.
        if self.course_near_goal and not orbit_ready:
            self.course_recenter_active = False
            return self._course_goal_seek_control(
                self.course_goal_x if observed_goal_x is not None else None,
                width,
                now,
            )
        decision.target_x = camera_center
        decision.x_command = 0.0
        decision.y_command = self.course_forward_command
        decision.status = "COURSE CLEAR - ADVANCING"

        # Once every cone is clear, use encoder odometry to return to the
        # lateral line on which Course mode was armed. Vision alone cannot know
        # that an earlier avoidance strafe moved the whole robot off-center.
        odometry = self.odometry.snapshot()
        self.course_center_error_m = (
            self.course_center_x_m - odometry.x_m if odometry.valid else 0.0
        )
        if not decision.forward_corridor_blocked and odometry.valid:
            center_error_abs = abs(self.course_center_error_m)
            if (self.course_recenter_active and
                    center_error_abs <= COURSE_RECENTER_STOP_ERROR_M):
                self.course_recenter_active = False
            elif (not self.course_recenter_active and
                  center_error_abs >= COURSE_RECENTER_START_ERROR_M):
                self.course_recenter_active = True
        else:
            self.course_recenter_active = False

        if self.course_recenter_active:
            recenter_magnitude = max(
                COURSE_RECENTER_MIN_COMMAND,
                min(
                    COURSE_RECENTER_MAX_COMMAND,
                    abs(self.course_center_error_m) * COURSE_RECENTER_KP,
                ),
            )
            field_x = math.copysign(
                recenter_magnitude, self.course_center_error_m
            )
            field_y = min(
                decision.y_command, COURSE_RECENTER_FORWARD_COMMAND
            )
            recenter_heading = max(
                -COURSE_RECENTER_MAX_HEADING_DEG,
                min(
                    COURSE_RECENTER_MAX_HEADING_DEG,
                    self.course_center_error_m
                    * COURSE_RECENTER_HEADING_KP_DEG_PER_M,
                ),
            )
            heading = (
                self.actual_heading
                if self.actual_heading is not None else recenter_heading
            )
            robot_x, robot_y = self._field_vector_to_robot(
                field_x, field_y, heading
            )
            self.commanded_heading = recenter_heading
            self.course_target_x = camera_center
            self.course_x_command = robot_x
            self.course_y_command = robot_y
            self.course_nearest_cone_ratio = decision.nearest_bottom_ratio
            direction = "RIGHT" if field_x > 0.0 else "LEFT"
            self.course_status = (
                f"COURSE RECENTER {direction} - LOOKING INWARD"
            )
            self._send_motion(
                f"CTRL,{robot_x:.1f},{robot_y:.1f},{recenter_heading:.2f}"
            )
            return self.course_status
        if self.course_phase == "RECENTER":
            self.course_phase = "CRUISE"
            self.commanded_heading = 0.0
            self.course_target_x = camera_center
            self.course_x_command = 0.0
            self.course_y_command = self.course_forward_command
            self.course_status = "COURSE CENTERED - FORWARD"
            self._send_motion(
                f"CTRL,0.0,{self.course_forward_command:.1f},0.00"
            )
            return self.course_status
        elif not cones and odometry.valid:
            # On a clear course the saved physical centerline, not a noisy or
            # partly occluded goal centroid, owns lateral motion.
            decision.x_command = 0.0
            decision.target_x = camera_center
            if decision.status == "COURSE CLEAR - ADVANCING":
                decision.status = "COURSE CENTERED - ADVANCING"
        self.course_target_x = decision.target_x
        self.course_x_command = decision.x_command
        self.course_y_command = decision.y_command
        self.course_nearest_cone_ratio = decision.nearest_bottom_ratio
        self.course_status = decision.status
        self._send_motion(
            f"CTRL,{decision.x_command:.1f},{decision.y_command:.1f},0.00"
        )
        return decision.status

    def _waypoint_control(self) -> str:
        self._set_dribbler(False)
        if self.waypoint_target_x is None or self.waypoint_target_y is None:
            self.waypoint_x_command = 0.0
            self.waypoint_y_command = 0.0
            self._send_motion("CTRL,0.0,0.0,0.00")
            self.waypoint_status = "WAYPOINT MISSING - STOPPED"
            return self.waypoint_status

        odometry = self.odometry.snapshot()
        if not odometry.valid:
            self.waypoint_x_command = 0.0
            self.waypoint_y_command = 0.0
            self.waypoint_arrived_frames = 0
            self._send_motion("CTRL,0.0,0.0,0.00")
            self.waypoint_status = "ODOMETRY LOST - STOPPED"
            return self.waypoint_status

        decision = plan_waypoint_motion(
            odometry.x_m,
            odometry.y_m,
            odometry.heading_deg,
            self.waypoint_target_x,
            self.waypoint_target_y,
            self.waypoint_max_command,
            min(WAYPOINT_MIN_COMMAND, self.waypoint_max_command),
            WAYPOINT_SLOW_RADIUS_M,
            self.waypoint_arrival_radius_m,
        )
        self.waypoint_distance_m = decision.distance_m
        self.commanded_heading = 0.0
        if decision.arrived:
            self.waypoint_arrived_frames += 1
            self.waypoint_x_command = 0.0
            self.waypoint_y_command = 0.0
            self._send_motion("CTRL,0.0,0.0,0.00")
            self.waypoint_status = (
                "WAYPOINT ARRIVED"
                if self.waypoint_arrived_frames >= WAYPOINT_ARRIVAL_CONFIRM_FRAMES
                else "WAYPOINT SETTLING"
            )
            return self.waypoint_status

        self.waypoint_arrived_frames = 0
        self.waypoint_x_command = decision.x_command
        self.waypoint_y_command = decision.y_command
        self._send_motion(
            f"CTRL,{decision.x_command:.1f},{decision.y_command:.1f},0.00"
        )
        self.waypoint_status = "WAYPOINT TRAVELLING"
        return self.waypoint_status

    def _auto_control(self, raw_error: float, ball_bottom: float,
                      width: int, now: float) -> str:
        self._set_dribbler(True)
        self.last_ball_bottom = ball_bottom
        self.last_ball_error = raw_error

        if self.control_mode == "HOLDING":
            # A clearly visible, non-close ball means the timed capture probably
            # failed or the ball escaped; resume normal acquisition.
            if ball_bottom < self._capture_lost_row_ratio():
                self.control_mode = "ALIGN"
                self.alignment_samples.clear()
                self.align_settle_until = now + ALIGN_SETTLE_TIME_S
            else:
                self._send_motion(
                    f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
                )
                return "HOLDING BALL"

        if self.control_mode == "CAPTURE_PUSH":
            return self._capture_control(now)

        if self.control_mode in {"PRE_SEARCH", "SEARCH"}:
            # Cancel the sweep at the measured chassis angle before estimating
            # the ball's bearing from fresh, settled camera frames.
            if self.actual_heading is not None:
                self.commanded_heading = self.actual_heading
            self.control_mode = "ALIGN"
            self.waiting_for_heading = False
            self.alignment_samples.clear()
            self._reset_near_strafe()
            self.align_settle_until = now + ALIGN_SETTLE_TIME_S
            self._send_motion(
                f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
            )
            return "BALL ACQUIRED"

        status = self.control_mode
        if self.control_mode == "ALIGN":
            self.offcenter_frames = 0
            if self.waiting_for_heading:
                heading_reached = False
                if self.actual_heading is not None:
                    heading_error = (
                        (self.commanded_heading - self.actual_heading + 180.0) % 360.0
                    ) - 180.0
                    heading_reached = (
                        abs(heading_error) <= HEADING_REACHED_TOLERANCE_DEG
                    )
                if heading_reached or now >= self.heading_move_deadline:
                    self.waiting_for_heading = False
                    self.align_settle_until = now + ALIGN_SETTLE_TIME_S
                    status = "SETTLING"
                else:
                    status = "TURNING"
            elif now < self.align_settle_until:
                self.alignment_samples.clear()
                status = "SETTLING"
            else:
                self.alignment_samples.append(raw_error)
                status = "MEASURING"
                if len(self.alignment_samples) >= ALIGN_MEASUREMENT_FRAMES:
                    measured = sum(self.alignment_samples) / len(self.alignment_samples)
                    self.alignment_samples.clear()
                    if abs(measured) <= CENTER_TOLERANCE_PX:
                        self.control_mode = "APPROACH"
                        status = "APPROACH"
                    else:
                        bearing = (
                            measured / (width / 2.0)
                        ) * (CAMERA_HORIZONTAL_FOV_DEG / 2.0)
                        bearing = max(
                            -MAX_DIRECT_BEARING_DEG,
                            min(MAX_DIRECT_BEARING_DEG, bearing),
                        )
                        self.commanded_heading += (
                            CAMERA_TO_HEADING_SIGN * bearing
                        )
                        self.commanded_heading = (
                            (self.commanded_heading + 180.0) % 360.0
                        ) - 180.0
                        self.waiting_for_heading = True
                        self.heading_move_deadline = now + HEADING_MOVE_TIMEOUT_S
                        status = "TURNING"
        elif (self.control_mode == "APPROACH" and
              ball_bottom >= self.reference_row_ratio):
            # Close to the ball, preserve chassis orientation and use X strafe
            # instead of repeatedly changing the gyro target.
            self.offcenter_frames = 0
        else:
            self.offcenter_frames = (
                self.offcenter_frames + 1
                if abs(raw_error) > CENTER_RELEASE_PX else 0
            )
            if self.offcenter_frames >= OFFCENTER_FRAMES_TO_REALIGN:
                self.control_mode = "ALIGN"
                self.offcenter_frames = 0
                self.alignment_samples.clear()
                self.align_settle_until = now + ALIGN_SETTLE_TIME_S
                self.waiting_for_heading = False
                status = "REALIGN"

        x_command = 0.0
        y_command = 0.0
        if self.control_mode == "APPROACH":
            if ball_bottom < self.reference_row_ratio:
                self._reset_near_strafe()
                y_command = FAR_FORWARD_COMMAND
            else:
                x_command, y_command, status = self._near_strafe_commands(
                    raw_error,
                    stop_forward=(ball_bottom >= self.capture_row_ratio),
                )
                if (ball_bottom >= self.capture_row_ratio and
                        self.near_center_latched):
                    self._start_capture(now)
                    return self._capture_control(now)
        self._send_motion(
            f"CTRL,{x_command:.1f},{y_command:.1f},"
            f"{self.commanded_heading:.2f}"
        )
        return status

    def _reset_near_strafe(self) -> None:
        self.near_error_filtered = 0.0
        self.near_filter_initialized = False
        self.near_center_latched = False
        self.near_x_command = 0.0

    def _near_strafe_commands(self, raw_error: float,
                              stop_forward: bool) -> tuple[float, float, str]:
        if not self.near_filter_initialized:
            self.near_error_filtered = raw_error
            self.near_filter_initialized = True
        else:
            self.near_error_filtered += NEAR_ERROR_EMA_ALPHA * (
                raw_error - self.near_error_filtered
            )

        error = self.near_error_filtered
        if self.near_center_latched:
            if abs(error) > NEAR_STRAFE_RELEASE_PX:
                self.near_center_latched = False
        elif abs(error) <= NEAR_STRAFE_CENTER_PX:
            self.near_center_latched = True

        if self.near_center_latched:
            desired_x = 0.0
            y_command = NEAR_CENTERED_FORWARD_COMMAND
            status = "NEAR CENTERED"
        else:
            scale = min(1.0, abs(error) / NEAR_STRAFE_FULL_ERROR_PX)
            magnitude = NEAR_STRAFE_MIN_COMMAND + (
                NEAR_STRAFE_MAX_COMMAND - NEAR_STRAFE_MIN_COMMAND
            ) * scale
            desired_x = math.copysign(magnitude, error)
            y_command = (
                0.0 if abs(error) >= NEAR_STRAFE_FULL_ERROR_PX
                else NEAR_STRAFE_FORWARD_COMMAND
            )
            status = "NEAR STRAFE"

        if stop_forward and not self.near_center_latched:
            y_command = 0.0

        # Stop immediately at center or before changing strafe direction. Limit
        # acceleration away from zero to avoid the earlier left-right violence.
        if desired_x == 0.0:
            self.near_x_command = 0.0
        elif self.near_x_command * desired_x < 0.0:
            self.near_x_command = 0.0
        else:
            delta = desired_x - self.near_x_command
            delta = max(
                -NEAR_STRAFE_ACCEL_PER_FRAME,
                min(NEAR_STRAFE_ACCEL_PER_FRAME, delta),
            )
            self.near_x_command += delta

        return self.near_x_command, y_command, status

    def _start_capture(self, now: float) -> None:
        self.control_mode = "CAPTURE_PUSH"
        self.capture_until = now + self.capture_push_time_s
        self._set_dribbler(True)

    def _capture_control(self, now: float) -> str:
        if now < self.capture_until:
            self._send_motion(
                f"CTRL,0.0,{CAPTURE_PUSH_COMMAND:.1f},"
                f"{self.commanded_heading:.2f}"
            )
            return "CAPTURE PUSH"
        self.control_mode = "GOAL_SEARCH"
        self.goal_search_last_time = now
        self.goal_seen_frames = 0
        self.goal_lost_frames = 0
        if self.actual_heading is not None:
            self.commanded_heading = self.actual_heading
        return self._goal_search_motion(now)

    @staticmethod
    def _is_goal_phase(state: str) -> bool:
        return state in {
            "GOAL_SEARCH",
            "GOAL_ALIGN",
            "GOAL_APPROACH",
            "GOAL_BACKOFF",
            "SHOOT_READY",
            "SHOOT_SETTLE",
            "POST_KICK",
        }

    def _goal_search_motion(self, now: float) -> str:
        dt = max(0.0, min(0.12, now - self.goal_search_last_time))
        self.goal_search_last_time = now
        # Negative heading is the verified physical left direction. Slow Y plus
        # a changing heading makes a rolling circle and keeps the ball pressed
        # into the front dribbler instead of spinning on the spot.
        self.commanded_heading = self._wrap_heading(
            self.commanded_heading - GOAL_SEARCH_HEADING_RATE_DEG_S * dt
        )
        self._set_dribbler(True)
        self._send_motion(
            f"CTRL,0.0,{GOAL_SEARCH_FORWARD_COMMAND:.1f},"
            f"{self.commanded_heading:.2f}"
        )
        return "GOAL SEARCH - ROLLING LEFT"

    def _side_safety_blocks_shot(self) -> bool:
        return bool(
            self.side_avoidance_enabled
            and (
                not self.side_camera_connected
                or self.side_left_obstacle
                or self.side_right_obstacle
            )
        )

    def _goal_control(self, error: float, bottom_ratio: float, width: int,
                      confidence: float, now: float) -> str:
        self._set_dribbler(True)
        self.goal_error_px = error
        self.goal_bottom_ratio = bottom_ratio
        self.goal_confidence = confidence

        if self.control_mode == "POST_KICK":
            if now < self.goal_post_kick_until:
                self._send_motion(
                    f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
                )
                return "KICKED"
            self._set_dribbler(False)
            self.control_mode = "ALIGN"
            self.target = None
            self.lost_frames = LOST_FRAMES_BEFORE_SEARCH + 1
            self.alignment_samples.clear()
            self.waiting_for_heading = False
            return "RETURNING TO BALL SEARCH"

        if self.control_mode == "SHOOT_SETTLE":
            if self._side_safety_blocks_shot():
                self.control_mode = "GOAL_APPROACH"
                self.goal_kick_sent = False
                self.goal_centered_frames = 0
                self._send_motion(
                    f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
                )
                return f"SHOT CANCELLED - {self.side_avoidance_state}"
            self._send_motion(f"CTRL,0.0,0.0,{self.commanded_heading:.2f}")
            if now >= self.goal_shoot_settle_until and not self.goal_kick_sent:
                self._send("KICK")
                self.goal_kick_sent = True
                self.goal_post_kick_until = now + GOAL_POST_KICK_TIME_S
                self.control_mode = "POST_KICK"
                return "KICKED"
            return "SETTLING TO SHOOT"

        if self.control_mode == "GOAL_SEARCH":
            if self.goal_seen_frames < GOAL_LOCK_FRAMES:
                return self._goal_search_motion(now)
            self.control_mode = "GOAL_ALIGN"
            self.goal_bearing_initialized = False
            self.goal_centered_frames = 0

        bearing = (error / (width / 2.0)) * (CAMERA_HORIZONTAL_FOV_DEG / 2.0)
        bearing = max(-GOAL_MAX_BEARING_DEG, min(GOAL_MAX_BEARING_DEG, bearing))
        if not self.goal_bearing_initialized:
            self.goal_bearing_filtered = bearing
            self.goal_bearing_initialized = True
        else:
            self.goal_bearing_filtered += GOAL_BEARING_EMA_ALPHA * (
                bearing - self.goal_bearing_filtered
            )

        base_heading = (
            self.actual_heading
            if self.actual_heading is not None else self.commanded_heading
        )
        self.commanded_heading = self._wrap_heading(
            base_heading + CAMERA_TO_HEADING_SIGN * self.goal_bearing_filtered
        )

        if abs(error) <= GOAL_CENTER_TOLERANCE_PX:
            self.goal_centered_frames += 1
        else:
            self.goal_centered_frames = 0

        at_shoot_distance = bottom_ratio >= self.goal_shoot_row_ratio
        remaining_image = max(0.0, 1.0 - self.goal_shoot_row_ratio)
        too_close_enter_row = self.goal_shoot_row_ratio + min(
            GOAL_TOO_CLOSE_ENTER_MARGIN_RATIO,
            remaining_image * 0.75,
        )
        backoff_exit_row = self.goal_shoot_row_ratio + min(
            GOAL_TOO_CLOSE_EXIT_MARGIN_RATIO,
            remaining_image * 0.25,
        )
        too_close_enter = bottom_ratio >= too_close_enter_row
        backoff_cleared = bottom_ratio <= backoff_exit_row
        centered = self.goal_centered_frames >= GOAL_CENTERED_FRAMES_TO_SHOOT
        if self._side_safety_blocks_shot():
            self.goal_centered_frames = 0
            self._send_motion(f"CTRL,0.0,0.0,{self.commanded_heading:.2f}")
            return f"SIDE AVOIDANCE - {self.side_avoidance_state}"

        # Distance correction takes priority over shooting. Keep using the
        # camera-derived gyro target while reversing so the robot backs away
        # without abandoning goal alignment. Separate enter/exit thresholds
        # avoid oscillating around the shoot line.
        if self.control_mode == "GOAL_BACKOFF":
            if not backoff_cleared:
                self.goal_centered_frames = 0
                self._send_motion(
                    f"CTRL,0.0,{-GOAL_BACKOFF_REVERSE_COMMAND:.1f},"
                    f"{self.commanded_heading:.2f}"
                )
                return "GOAL TOO CLOSE - REVERSING"
            self.control_mode = "GOAL_APPROACH"
            self.goal_centered_frames = 0
        elif too_close_enter:
            self.control_mode = "GOAL_BACKOFF"
            self.goal_centered_frames = 0
            self._send_motion(
                f"CTRL,0.0,{-GOAL_BACKOFF_REVERSE_COMMAND:.1f},"
                f"{self.commanded_heading:.2f}"
            )
            return "GOAL TOO CLOSE - REVERSING"

        if at_shoot_distance and centered:
            self._send_motion(f"CTRL,0.0,0.0,{self.commanded_heading:.2f}")
            if not self.auto_kick_enabled:
                self.control_mode = "SHOOT_READY"
                return "SHOOT READY - AUTO KICK OFF"
            self.control_mode = "SHOOT_SETTLE"
            self.goal_shoot_settle_until = now + GOAL_SHOOT_SETTLE_TIME_S
            self.goal_kick_sent = False
            return "SETTLING TO SHOOT"

        if self.control_mode == "SHOOT_READY":
            if not at_shoot_distance or abs(error) > GOAL_CENTER_RELEASE_PX:
                self.control_mode = "GOAL_APPROACH"
            else:
                self._send_motion(
                    f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
                )
                return "SHOOT READY - AUTO KICK OFF"

        if abs(error) > GOAL_CENTER_RELEASE_PX:
            self.control_mode = "GOAL_ALIGN"
            forward = GOAL_ALIGN_FORWARD_COMMAND
            status = "GOAL ALIGNING ARC"
        else:
            self.control_mode = "GOAL_APPROACH"
            forward = (
                GOAL_ALIGN_FORWARD_COMMAND
                if at_shoot_distance else GOAL_APPROACH_FORWARD_COMMAND
            )
            status = "GOAL APPROACH"
        self._send_motion(
            f"CTRL,0.0,{forward:.1f},{self.commanded_heading:.2f}"
        )
        return status

    def _goal_without_target(self, now: float) -> str:
        if self.control_mode == "POST_KICK":
            return self._goal_control(
                0.0, self.goal_bottom_ratio, 640, self.goal_confidence, now
            )
        if self.control_mode == "SHOOT_SETTLE":
            if self._side_safety_blocks_shot():
                self.control_mode = "GOAL_SEARCH"
                self.goal_kick_sent = False
                self.goal_search_last_time = now
                return f"SHOT CANCELLED - {self.side_avoidance_state}"
            self._send_motion(f"CTRL,0.0,0.0,{self.commanded_heading:.2f}")
            if now >= self.goal_shoot_settle_until and not self.goal_kick_sent:
                self._send("KICK")
                self.goal_kick_sent = True
                self.goal_post_kick_until = now + GOAL_POST_KICK_TIME_S
                self.control_mode = "POST_KICK"
                return "KICKED"
            return "SETTLING TO SHOOT"
        if (self.control_mode == "GOAL_BACKOFF" and
                self.goal_lost_frames <= GOAL_LOST_FRAMES_BEFORE_SEARCH):
            # Do not reverse blind when a very close goal fills or leaves the
            # image. Hold position briefly and resume search if loss persists.
            self._send_motion(
                f"CTRL,0.0,0.0,{self.commanded_heading:.2f}"
            )
            return "CLOSE GOAL LOST - HOLDING"
        if self.goal_lost_frames <= GOAL_LOST_FRAMES_BEFORE_SEARCH:
            self._send_motion(
                f"CTRL,0.0,{GOAL_ALIGN_FORWARD_COMMAND:.1f},"
                f"{self.commanded_heading:.2f}"
            )
            return "GOAL LOST - CONTINUING ARC"
        if self.control_mode != "GOAL_SEARCH":
            self.control_mode = "GOAL_SEARCH"
            self.goal_search_last_time = now
            if self.actual_heading is not None:
                self.commanded_heading = self.actual_heading
            self.goal_seen_frames = 0
            self.goal_bearing_initialized = False
        return self._goal_search_motion(now)

    def _auto_without_target(self, now: float) -> str:
        if self.control_mode == "CAPTURE_PUSH":
            return self._capture_control(now)
        if self.control_mode == "HOLDING":
            self._send_motion(f"CTRL,0.0,0.0,{self.commanded_heading:.2f}")
            return "HOLDING BALL"

        # A close ball normally disappears under the camera as the dribbler
        # takes it. Convert that expected loss into the short capture push.
        if (self.control_mode == "APPROACH" and
                self.last_ball_bottom >= self._capture_lost_row_ratio() and
                abs(self.last_ball_error) <= CAPTURE_CENTER_TOLERANCE_PX and
                self.lost_frames >= CAPTURE_LOST_FRAMES):
            self._start_capture(now)
            return self._capture_control(now)

        if self.lost_frames <= LOST_FRAMES_BEFORE_SEARCH:
            self._send_motion(f"CTRL,0.0,0.0,{self.commanded_heading:.2f}")
            return "TARGET LOST - WAITING"

        if self.control_mode not in {"PRE_SEARCH", "SEARCH"}:
            self.control_mode = "PRE_SEARCH"
            self._reset_near_strafe()
            self.search_center_heading = (
                self.actual_heading
                if self.actual_heading is not None else self.commanded_heading
            )
            self.search_prescan_index = 0
            self.search_swept_degrees = 0.0
            self.commanded_heading = self._wrap_heading(
                self.search_center_heading +
                SEARCH_PRE_SCAN_OFFSETS_DEG[self.search_prescan_index]
            )
            self.search_leg_deadline = now + SEARCH_PRE_SCAN_LEG_TIMEOUT_S

        elif self.control_mode == "PRE_SEARCH":
            reached = False
            if self.actual_heading is not None:
                error = self._wrap_heading(
                    self.commanded_heading - self.actual_heading
                )
                reached = abs(error) <= SEARCH_HEADING_TOLERANCE_DEG
            if reached or now >= self.search_leg_deadline:
                self.search_prescan_index += 1
                if self.search_prescan_index < len(SEARCH_PRE_SCAN_OFFSETS_DEG):
                    self.commanded_heading = self._wrap_heading(
                        self.search_center_heading
                        + SEARCH_PRE_SCAN_OFFSETS_DEG[self.search_prescan_index]
                    )
                    self.search_leg_deadline = (
                        now + SEARCH_PRE_SCAN_LEG_TIMEOUT_S
                    )
                else:
                    # Nothing nearby: start the existing full sweep to the
                    # left, then reverse direction after each 360 degrees.
                    self.control_mode = "SEARCH"
                    self.search_direction = -1.0
                    self.search_swept_degrees = 0.0
                    self.commanded_heading = self._wrap_heading(
                        self.search_center_heading
                        + self.search_direction * SEARCH_STEP_ANGLE_DEG
                    )
                    self.search_leg_deadline = now + SEARCH_LEG_TIMEOUT_S

        else:
            reached = False
            if self.actual_heading is not None:
                error = self._wrap_heading(
                    self.commanded_heading - self.actual_heading
                )
                reached = abs(error) <= SEARCH_HEADING_TOLERANCE_DEG
            if reached or now >= self.search_leg_deadline:
                self.search_swept_degrees += SEARCH_STEP_ANGLE_DEG
                if self.search_swept_degrees >= SEARCH_FULL_ROTATION_DEG:
                    self.search_direction *= -1.0
                    self.search_swept_degrees = 0.0
                self.commanded_heading = self._wrap_heading(
                    self.commanded_heading +
                    self.search_direction * SEARCH_STEP_ANGLE_DEG
                )
                self.search_leg_deadline = now + SEARCH_LEG_TIMEOUT_S

        self._send_motion(f"CTRL,0.0,0.0,{self.commanded_heading:.2f}")
        if self.control_mode == "PRE_SEARCH":
            labels = ("RIGHT", "LEFT", "CENTER")
            return f"PRE-SCAN {labels[self.search_prescan_index]}"
        direction = "LEFT" if self.search_direction < 0.0 else "RIGHT"
        return (
            f"SEARCH {direction} "
            f"{round(self.search_swept_degrees):d}/"
            f"{round(SEARCH_FULL_ROTATION_DEG):d} DEG"
        )

    def _read_serial(self) -> None:
        if self.serial is None:
            return
        try:
            with self.serial_lock:
                while self.serial.in_waiting:
                    line = self.serial.readline().decode(
                        "utf-8", errors="replace"
                    ).strip()
                    if line and line not in {"OK,MOVE", "OK,CTRL"}:
                        if not line.startswith("STATUS,"):
                            self.last_serial_line = line
                        if line.startswith("STATUS,"):
                            heading_match = re.search(r"(?:^|,)H=(-?\d+(?:\.\d+)?)", line)
                            imu_match = re.search(r"(?:^|,)IMU=(\d+)", line)
                            calibration_match = re.search(r"(?:^|,)CAL=(\d+)", line)
                            progress_match = re.search(r"(?:^|,)CALP=(\d+)", line)
                            ms_match = re.search(r"(?:^|,)MS=(\d+)", line)
                            rpm1_match = re.search(
                                r"(?:^|,)RPM1=(-?\d+(?:\.\d+)?)", line
                            )
                            rpm2_match = re.search(
                                r"(?:^|,)RPM2=(-?\d+(?:\.\d+)?)", line
                            )
                            rpm3_match = re.search(
                                r"(?:^|,)RPM3=(-?\d+(?:\.\d+)?)", line
                            )
                            if heading_match:
                                self.actual_heading = float(heading_match.group(1))
                            if imu_match:
                                self.imu_ok = int(imu_match.group(1))
                            if progress_match:
                                self.calibration_progress = max(
                                    0, min(100, int(progress_match.group(1)))
                                )
                            if calibration_match:
                                reported_active = bool(int(calibration_match.group(1)))
                                if reported_active:
                                    self.calibrating = True
                                elif (self.calibration_requested and
                                      self.calibration_progress >= 100):
                                    self.calibrating = False
                                    self.calibration_requested = False
                                    self.calibration_complete_until = (
                                        time.monotonic() + 4.0
                                    )
                            if (ms_match and heading_match and rpm1_match and
                                    rpm2_match and rpm3_match):
                                with self.lock:
                                    self.odometry.update(
                                        int(ms_match.group(1)),
                                        float(rpm1_match.group(1)),
                                        float(rpm2_match.group(1)),
                                        float(rpm3_match.group(1)),
                                        float(heading_match.group(1)),
                                        self.odometry_x_scale,
                                        self.odometry_y_scale,
                                    )
        except serial.SerialException:
            pass

    def _run_side_camera(self) -> None:
        assert self.side_capture is not None
        while self.running:
            ok, frame = self.side_capture.read()
            if not ok:
                with self.lock:
                    self.side_camera_connected = False
                    self.side_avoidance_state = "SAFETY CAMERA OFFLINE"
                time.sleep(0.05)
                continue
            height, width = frame.shape[:2]
            with self.lock:
                values = dict(self.side_mask)
                self.side_camera_connected = True
                floor_model = self.floor_model
                floor_sample_active = self.floor_calibration_collecting
                floor_sample_center = self.floor_calibration_center

            top = round(height * values["top"] / 100.0)
            bottom = round(height * values["bottom"] / 100.0)
            left_outer = round(width * values["left_outer"] / 100.0)
            left_inner = round(width * values["left_inner"] / 100.0)
            right_inner = round(width * values["right_inner"] / 100.0)
            right_outer = round(width * values["right_outer"] / 100.0)

            active = np.zeros((height, width), dtype=np.uint8)
            active[top:bottom, left_outer:left_inner] = 255
            active[top:bottom, right_inner:right_outer] = 255

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            self._collect_floor_calibration(lab, time.monotonic())
            left_result = right_result = None
            if floor_model is not None:
                foreign = foreign_pixel_mask(lab, floor_model)
                left_result, _left_clean = analyze_zone(
                    foreign, (left_outer, top, left_inner, bottom), floor_model
                )
                right_result, _right_clean = analyze_zone(
                    foreign, (right_inner, top, right_outer, bottom), floor_model
                )
                self._update_side_obstacles(
                    left_result.detected, right_result.detected
                )
                with self.lock:
                    self.side_left_area_ratio = left_result.area_ratio
                    self.side_right_area_ratio = right_result.area_ratio
            else:
                self._update_side_obstacles(False, False)

            inactive = active == 0
            tinted = frame.astype(np.float32) * 0.24
            tinted[:, :, 2] += 55.0
            frame[inactive] = np.clip(tinted[inactive], 0, 255).astype(np.uint8)

            cv2.rectangle(frame, (left_outer, top), (left_inner, bottom),
                          (0, 255, 0), 2)
            cv2.rectangle(frame, (right_inner, top), (right_outer, bottom),
                          (0, 255, 0), 2)
            cv2.putText(frame, "LEFT ACTIVE ZONE", (left_outer + 8, top + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
            cv2.putText(frame, "RIGHT ACTIVE ZONE", (right_inner + 8, top + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
            if left_result is not None and left_result.detected and left_result.bbox:
                x, y, box_width, box_height = left_result.bbox
                cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                              (0, 0, 255), 3)
                cv2.putText(frame, "FOREIGN OBJECT", (x, max(18, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
            if right_result is not None and right_result.detected and right_result.bbox:
                x, y, box_width, box_height = right_result.bbox
                cv2.rectangle(frame, (x, y), (x + box_width, y + box_height),
                              (0, 0, 255), 3)
                cv2.putText(frame, "FOREIGN OBJECT", (x, max(18, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
            if floor_sample_active:
                cv2.drawMarker(frame, floor_sample_center, (255, 0, 255),
                               cv2.MARKER_CROSS, 28, 2)
                cv2.circle(frame, floor_sample_center,
                           FLOOR_CALIBRATION_PATCH_RADIUS_PX,
                           (255, 0, 255), 1)
            with self.lock:
                avoidance_state = self.side_avoidance_state
            cv2.putText(frame, avoidance_state, (12, height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (0, 165, 255) if "CLEAR" in avoidance_state else (0, 0, 255),
                        2)

            encoded, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72]
            )
            if encoded:
                with self.lock:
                    self.side_frame_jpeg = jpeg.tobytes()

    def _run(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            height, width = frame.shape[:2]
            with self.lock:
                selected = self.selected_color
                mode = self.mode
                display_view = self.display_view
                reference_row_ratio = self.reference_row_ratio
                capture_row_ratio = self.capture_row_ratio
                goal_shoot_row_ratio = self.goal_shoot_row_ratio
                ball_sample_active = self.ball_calibration_collecting
                ball_sample_center = self.ball_calibration_center
                goal_selected = self.selected_goal_color
                goal_sample_active = self.goal_calibration_collecting
                goal_sample_box = self.goal_calibration_box
                course_value_max = self.course_black_value_max
                course_roi_top = self.course_roi_top_ratio
                course_min_area = self.course_min_cone_area
                course_side_danger_width = self.course_side_danger_width_ratio
                course_orbit_start = self.course_orbit_start_ratio
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sample_now = time.monotonic()
            self._collect_ball_calibration(hsv, sample_now)
            self._collect_goal_calibration(hsv, sample_now)
            mask = hue_mask(hsv, self.filters[selected])
            candidates = find_candidates(mask, self.filters[selected])

            goal_candidates, _goal_mask = find_goal_candidates(
                hsv, self.goal_filters[goal_selected]
            )
            now = time.monotonic()
            if mode in {"safe", "course"}:
                detected_cones, _cone_mask = detect_black_cones(
                    hsv, course_value_max, course_roi_top, course_min_area
                )
                with self.lock:
                    tracked_obstacles = self.course_obstacle_tracker.update(
                        detected_cones
                    )
                    if tracked_obstacles:
                        self.course_cones = tracked_obstacles
                        self.course_last_cone_time = now
                    elif now - self.course_last_cone_time > COURSE_CONE_HOLD_TIME_S:
                        self.course_cones = []
                    course_cones = list(self.course_cones)
                    self.course_nearest_cone_ratio = max(
                        (item.bottom / height for item in course_cones),
                        default=0.0,
                    )
            else:
                course_cones = []
            selected_goal = select_locked_goal(goal_candidates, self.goal_target)
            if selected_goal is None:
                self.goal_lost_frames += 1
                self.goal_seen_frames = 0
                if self.goal_lost_frames > GOAL_LOST_FRAMES_BEFORE_SEARCH:
                    self.goal_target = None
            else:
                self.goal_target = selected_goal
                self.goal_lost_frames = 0
                self.goal_seen_frames += 1

            # If the ball and goal use the same color, prevent the large goal
            # rectangle itself from winning the ball detector's area score.
            # A real ball visible inside the goal remains eligible when it is
            # small relative to the goal box.
            if selected == goal_selected and selected_goal is not None:
                gx, gy, gw, gh = selected_goal.bbox
                candidates = [
                    item for item in candidates
                    if not (
                        gx <= item.center[0] <= gx + gw
                        and gy <= item.center[1] <= gy + gh
                        and (
                            item.radius > 0.25 * gh
                            or item.area > 0.18 * selected_goal.area
                        )
                    )
                ]
            selected_target = select_locked_target(candidates, self.target)
            if selected_target is None:
                self.lost_frames += 1
                if self.lost_frames > 8:
                    self.target = None
            else:
                self.target = selected_target
                self.lost_frames = 0

            target_x = round(width * TARGET_X_RATIO)
            ref_y = round(height * reference_row_ratio)
            capture_y = round(height * capture_row_ratio)
            goal_shoot_y = round(height * goal_shoot_row_ratio)
            cv2.line(frame, (target_x, 0), (target_x, height), (120, 120, 120), 1)
            if display_view != "course":
                cv2.line(frame, (0, ref_y), (width, ref_y), (255, 180, 0), 1)
                cv2.line(frame, (0, capture_y), (width, capture_y), (255, 0, 255), 1)
                cv2.line(frame, (0, goal_shoot_y), (width, goal_shoot_y),
                         (0, 165, 255), 1)
                cv2.putText(frame, "near / strafe", (8, max(72, ref_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 180, 0), 1)
                cv2.putText(frame, "capture", (8, max(90, capture_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 0, 255), 1)
                cv2.putText(frame, "goal shoot", (max(8, width - 105),
                            max(108, goal_shoot_y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.43, (0, 165, 255), 1)
            if ball_sample_active:
                cv2.drawMarker(frame, ball_sample_center, (255, 0, 255),
                               cv2.MARKER_CROSS, 28, 2)
                cv2.circle(frame, ball_sample_center,
                           BALL_CALIBRATION_PATCH_RADIUS_PX,
                           (255, 0, 255), 1)
            if goal_sample_active:
                left, top, right, bottom = goal_sample_box
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 165, 255), 2)

            if mode in {"safe", "course"}:
                course_roi_y = round(height * course_roi_top)
                course_orbit_y = round(height * course_orbit_start)
                course_left_danger_x = round(
                    width * (0.5 - course_side_danger_width)
                )
                course_right_danger_x = round(
                    width * (0.5 + course_side_danger_width)
                )
                cv2.line(frame, (0, course_roi_y), (width, course_roi_y),
                         (90, 90, 255), 1)
                cv2.putText(frame, "cone detection ROI", (8, max(18, course_roi_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (90, 90, 255), 1)
                # The lower red rectangle is the projected robot corridor.
                # Dark objects wholly outside it remain tracked but do not
                # trigger avoidance merely because they are near an edge.
                cv2.rectangle(
                    frame, (course_left_danger_x, course_orbit_y),
                    (course_right_danger_x, height - 1), (0, 0, 255), 2,
                )
                cv2.putText(
                    frame, "COLLISION CORRIDOR",
                    (course_left_danger_x + 6, course_orbit_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1,
                )
                cv2.line(
                    frame, (0, course_orbit_y), (width, course_orbit_y),
                    (255, 0, 255), 2,
                )
                cv2.putText(
                    frame, "ORBIT START",
                    (8, max(18, course_orbit_y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1,
                )
                for cone in course_cones:
                    cx, cy, cw, ch = cone.bbox
                    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, "DARK OBSTACLE", (cx, max(18, cy - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 255), 2)

            goal_error = 0.0
            goal_bottom = 0.0
            if self.goal_target is not None and self.goal_lost_frames == 0:
                gx, gy, gw, gh = self.goal_target.bbox
                goal_error = float(self.goal_target.center[0] - target_x)
                goal_bottom = self.goal_target.bottom / height
                cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh),
                              (0, 165, 255), 2)
                cv2.drawMarker(frame, self.goal_target.center, (0, 165, 255),
                               cv2.MARKER_CROSS, 24, 2)
                cv2.putText(
                    frame,
                    f"GOAL {goal_selected.upper()} {self.goal_target.confidence:.0f}%",
                    (gx, max(18, gy - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 165, 255), 2,
                )
                with self.lock:
                    self.goal_error_px = goal_error
                    self.goal_bottom_ratio = goal_bottom
                    self.goal_confidence = self.goal_target.confidence
                    if not self._is_goal_phase(self.control_mode):
                        self.goal_status = "GOAL DETECTED"
            else:
                with self.lock:
                    self.goal_error_px = 0.0
                    self.goal_bottom_ratio = 0.0
                    self.goal_confidence = 0.0
                    if not self._is_goal_phase(self.control_mode):
                        self.goal_status = "GOAL NOT SEEN"

            status = "SEARCHING"
            error = 0.0
            with self.lock:
                goal_phase = self._is_goal_phase(self.control_mode)
            strict_course_goal = (
                self.goal_target is not None and self.goal_lost_frames == 0
            )
            partial_goal = None
            if mode == "course" and not strict_course_goal:
                partial_goal = partial_goal_from_mask(_goal_mask)
            course_goal_x = (
                float(self.goal_target.center[0])
                if strict_course_goal else (
                    partial_goal.center_x if partial_goal is not None else None
                )
            )
            course_goal_bottom = (
                self.goal_target.bottom / height
                if strict_course_goal else (
                    partial_goal.bottom / height if partial_goal is not None else 0.0
                )
            )
            if mode == "course":
                with self.lock:
                    status = self._course_control(
                        course_cones,
                        width,
                        height,
                        course_goal_x,
                        course_goal_bottom,
                        strict_course_goal,
                        now,
                    )
                    error = self.course_target_x - target_x
                cv2.line(
                    frame,
                    (round(self.course_target_x), course_roi_y),
                    (round(self.course_target_x), height),
                    (255, 255, 255),
                    2,
                )
            elif mode == "waypoint":
                with self.lock:
                    status = self._waypoint_control()
            elif mode == "auto" and goal_phase:
                error = goal_error
                with self.lock:
                    if self.goal_target is not None and self.goal_lost_frames == 0:
                        status = self._goal_control(
                            goal_error,
                            goal_bottom,
                            width,
                            self.goal_target.confidence,
                            now,
                        )
                    else:
                        status = self._goal_without_target(now)
                    self.goal_status = status
            elif self.target is not None and self.lost_frames == 0:
                x, y = self.target.center
                error = float(x - target_x)
                bottom = (y + self.target.radius) / height
                cv2.circle(frame, (x, y), round(self.target.radius), (0, 255, 0), 2)
                cv2.drawMarker(frame, (x, y), (0, 255, 0),
                               cv2.MARKER_CROSS, 18, 2)
                status = "LOCKED"
                with self.lock:
                    if self.mode == "auto":
                        status = self._auto_control(error, bottom, width, now)
                    elif self.mode == "goalkeeper":
                        status = self._goalkeeper_control(error)
            elif mode == "auto":
                with self.lock:
                    status = self._auto_without_target(now)
            elif mode == "goalkeeper":
                with self.lock:
                    status = self._goalkeeper_without_target()

            with self.lock:
                if self.mode == "manual":
                    self._manual_control(now)
                self.vision_status = status
                self.error_px = error
                mode = self.mode
            self._read_serial()
            if self.serial is not None and now - self.last_status_request >= 0.1:
                self._send("STATUS")
                self.last_status_request = now

            cv2.putText(frame,
                        f"BALL {selected.upper()} | GOAL {goal_selected.upper()} | "
                        f"{mode.upper()} | {status}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2)
            cv2.putText(frame,
                        f"error={error:+.0f}px gyro="
                        f"{self.actual_heading if self.actual_heading is not None else '--'} "
                        f"target={self.commanded_heading:+.1f}",
                        (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)
            with self.lock:
                calibrating = self.calibrating
                calibration_progress = self.calibration_progress
                calibration_complete = now < self.calibration_complete_until
            if calibrating:
                bar_left, bar_right = 40, width - 40
                bar_top, bar_bottom = height // 2 - 28, height // 2 + 28
                cv2.rectangle(frame, (bar_left - 12, bar_top - 48),
                              (bar_right + 12, bar_bottom + 18), (15, 15, 15), -1)
                cv2.putText(frame, f"CALIBRATING {calibration_progress}% - KEEP STILL",
                            (bar_left, bar_top - 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2)
                cv2.rectangle(frame, (bar_left, bar_top),
                              (bar_right, bar_bottom), (90, 90, 90), 2)
                fill_right = bar_left + round(
                    (bar_right - bar_left) * calibration_progress / 100.0
                )
                cv2.rectangle(frame, (bar_left + 2, bar_top + 2),
                              (fill_right, bar_bottom - 2), (0, 210, 255), -1)
            elif calibration_complete:
                cv2.rectangle(frame, (40, height // 2 - 38),
                              (width - 40, height // 2 + 38), (15, 15, 15), -1)
                cv2.putText(frame, "CALIBRATION COMPLETE",
                            (max(50, width // 2 - 185), height // 2 + 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            encoded, jpeg = cv2.imencode(".jpg", frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 78])
            if encoded:
                with self.lock:
                    self.frame_jpeg = jpeg.tobytes()

    def snapshot(self) -> dict:
        with self.lock:
            odometry = self.odometry.snapshot()
            return {
                "mode": self.mode,
                "color": self.selected_color,
                "vision": self.vision_status,
                "error_px": round(self.error_px, 1),
                "actual_heading": (
                    round(self.actual_heading, 2)
                    if self.actual_heading is not None else None
                ),
                "target_heading": round(self.commanded_heading, 2),
                "manual_heading": round(self.manual_heading, 2),
                "goalkeeper_x": round(self.goalkeeper_x_command, 1),
                "course_status": self.course_status,
                "course_cone_count": len(self.course_cones),
                "course_x": round(self.course_x_command, 1),
                "course_y": round(self.course_y_command, 1),
                "course_target_x": round(self.course_target_x, 1),
                "course_nearest_cone_percent": round(
                    self.course_nearest_cone_ratio * 100.0, 1
                ),
                "course_goal_remembered": self.course_goal_x is not None,
                "course_center_x_m": round(self.course_center_x_m, 3),
                "course_center_error_m": round(
                    self.course_center_error_m, 3
                ),
                "course_recenter_active": self.course_recenter_active,
                "course_black_value_max": self.course_black_value_max,
                "course_roi_top_percent": round(
                    self.course_roi_top_ratio * 100.0, 1
                ),
                "course_min_cone_area": round(self.course_min_cone_area, 0),
                "course_forward_command": round(
                    self.course_forward_command, 1
                ),
                "course_side_danger_width_percent": round(
                    self.course_side_danger_width_ratio * 100.0, 1
                ),
                "course_orbit_start_percent": round(
                    self.course_orbit_start_ratio * 100.0, 1
                ),
                "course_orbit_pass_distance_m": round(
                    self.course_orbit_pass_distance_m, 2
                ),
                "course_orbit_command": round(self.course_orbit_command, 1),
                "course_goal_search_progress_percent": round(
                    self.course_goal_search_progress * 100.0, 1
                ),
                "course_post_clear_forward_s": round(
                    self.course_post_clear_forward_s, 2
                ),
                "course_phase": self.course_phase,
                "course_progress_percent": round(
                    self.course_progress_ratio * 100.0, 1
                ),
                "course_near_goal": self.course_near_goal,
                "course_pass_side": (
                    "RIGHT" if self.course_pass_side > 0.0 else (
                        "LEFT" if self.course_pass_side < 0.0 else "NONE"
                    )
                ),
                "field_map": {
                    "width_m": round(self.field_width_m, 3),
                    "length_m": round(self.field_length_m, 3),
                    "odometry_x_scale": round(self.odometry_x_scale, 4),
                    "odometry_y_scale": round(self.odometry_y_scale, 4),
                    "waypoint_max_command": round(
                        self.waypoint_max_command, 1
                    ),
                    "waypoint_arrival_radius_m": round(
                        self.waypoint_arrival_radius_m, 3
                    ),
                },
                "waypoint": {
                    "target": (
                        None if self.waypoint_target_x is None else {
                            "x_m": round(self.waypoint_target_x, 4),
                            "y_m": round(self.waypoint_target_y, 4),
                        }
                    ),
                    "distance_m": round(self.waypoint_distance_m, 4),
                    "x_command": round(self.waypoint_x_command, 1),
                    "y_command": round(self.waypoint_y_command, 1),
                    "status": self.waypoint_status,
                },
                "odometry": {
                    "x_m": round(odometry.x_m, 4),
                    "y_m": round(odometry.y_m, 4),
                    "vx_mps": round(odometry.vx_mps, 4),
                    "vy_mps": round(odometry.vy_mps, 4),
                    "speed_mps": round(odometry.speed_mps, 4),
                    "heading_deg": round(odometry.heading_deg, 2),
                    "rpm1": round(odometry.rpm1, 1),
                    "rpm2": round(odometry.rpm2, 1),
                    "rpm3": round(odometry.rpm3, 1),
                    "telemetry_age_s": (
                        None if odometry.telemetry_age_s is None
                        else round(odometry.telemetry_age_s, 3)
                    ),
                    "valid": odometry.valid,
                    "trail": [
                        [round(x, 4), round(y, 4)]
                        for x, y in odometry.trail
                    ],
                },
                "imu_ok": self.imu_ok,
                "calibrating": self.calibrating,
                "calibration_progress": self.calibration_progress,
                "auto_state": self.control_mode,
                "dribbler_running": self.dribbler_running,
                "auto_dribbler_pwm": self.auto_dribbler_pwm,
                "capture_push_time_s": round(self.capture_push_time_s, 2),
                "capture_push_estimated_distance_cm": round(
                    CAPTURE_PUSH_ESTIMATED_DISTANCE_CM
                    * self.capture_push_time_s / CAPTURE_PUSH_TIME_S,
                    1,
                ),
                "near_line_percent": round(self.reference_row_ratio * 100.0, 1),
                "capture_line_percent": round(self.capture_row_ratio * 100.0, 1),
                "goal_color": self.selected_goal_color,
                "goal_status": self.goal_status,
                "goal_error_px": round(self.goal_error_px, 1),
                "goal_confidence": round(self.goal_confidence, 1),
                "goal_bottom_percent": round(self.goal_bottom_ratio * 100.0, 1),
                "goal_shoot_line_percent": round(
                    self.goal_shoot_row_ratio * 100.0, 1
                ),
                "auto_kick_enabled": self.auto_kick_enabled,
                "side_camera_connected": self.side_camera_connected,
                "side_camera_index": self.side_camera_index,
                "side_mask": dict(self.side_mask),
                "side_avoidance_enabled": self.side_avoidance_enabled,
                "side_avoidance_state": self.side_avoidance_state,
                "side_left_obstacle": self.side_left_obstacle,
                "side_right_obstacle": self.side_right_obstacle,
                "side_left_area_percent": round(
                    self.side_left_area_ratio * 100.0, 2
                ),
                "side_right_area_percent": round(
                    self.side_right_area_ratio * 100.0, 2
                ),
                "floor_calibrated": self.floor_model is not None,
                "floor_calibration": {
                    "session": self.floor_calibration_session,
                    "collecting": self.floor_calibration_collecting,
                    "samples": self.floor_calibration_points,
                    "frame": self.floor_calibration_frame_count,
                    "frames_total": FLOOR_CALIBRATION_FRAMES_PER_SAMPLE,
                    "message": self.floor_calibration_message,
                    "result": self.floor_calibration_result,
                },
                "ball_calibration": {
                    "session": self.ball_calibration_session,
                    "collecting": self.ball_calibration_collecting,
                    "color": (
                        self.ball_calibration_color
                        if self.ball_calibration_session
                        else self.selected_color
                    ),
                    "samples": self.ball_calibration_points,
                    "frame": self.ball_calibration_frame_count,
                    "frames_total": BALL_CALIBRATION_FRAMES_PER_SAMPLE,
                    "message": self.ball_calibration_message,
                    "result": self.ball_calibration_result,
                },
                "goal_calibration": {
                    "session": self.goal_calibration_session,
                    "collecting": self.goal_calibration_collecting,
                    "color": (
                        self.goal_calibration_color
                        if self.goal_calibration_session
                        else self.selected_goal_color
                    ),
                    "samples": self.goal_calibration_points,
                    "frame": self.goal_calibration_frame_count,
                    "frames_total": GOAL_CALIBRATION_FRAMES_PER_SAMPLE,
                    "message": self.goal_calibration_message,
                    "result": self.goal_calibration_result,
                },
                "serial": "connected" if self.serial else "camera only",
                "last_serial": self.last_serial_line,
            }

    def close(self) -> None:
        self.running = False
        self._send("STOP")
        self.worker.join(timeout=1.0)
        self.capture.release()
        if self.side_worker is not None:
            self.side_worker.join(timeout=1.0)
        if self.side_capture is not None:
            self.side_capture.release()
        if self.serial:
            self.serial.close()


def create_app(controller: SokaWebController) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("web_control.html")

    @app.get("/video")
    def video():
        def frames():
            while controller.running:
                with controller.lock:
                    jpeg = controller.frame_jpeg
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                           jpeg + b"\r\n")
                time.sleep(0.05)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/side-video")
    def side_video():
        def frames():
            while controller.running:
                with controller.lock:
                    jpeg = controller.side_frame_jpeg
                if jpeg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                           jpeg + b"\r\n")
                time.sleep(0.07)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def state():
        return jsonify(controller.snapshot())

    @app.post("/api/mode")
    def mode():
        try:
            controller.set_mode(request.get_json(force=True)["mode"])
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/view")
    def view():
        view_name = request.get_json(force=True).get("view", "auto")
        valid_views = {
            "auto", "course", "fieldMap", "goalkeeper", "manual",
            "calibration", "goalCalibration", "side",
        }
        if view_name not in valid_views:
            return jsonify({"error": "invalid view"}), 400
        with controller.lock:
            controller.display_view = view_name
        return jsonify({"view": view_name})

    @app.post("/api/color")
    def color():
        color_name = request.get_json(force=True)["color"].lower()
        if color_name not in controller.filters:
            return jsonify({"error": "invalid color"}), 400
        controller.set_mode("safe")
        with controller.lock:
            if (controller.ball_calibration_session and
                    controller.ball_calibration_color != color_name):
                controller._clear_ball_calibration(
                    "Color changed; press Start new calibration."
                )
            controller.selected_color = color_name
            if not controller.ball_calibration_session:
                controller.ball_calibration_color = color_name
            controller.target = None
        return jsonify(controller.snapshot())

    @app.post("/api/goal-color")
    def goal_color():
        color_name = request.get_json(force=True)["color"].lower()
        if color_name not in controller.goal_filters:
            return jsonify({"error": "invalid goal color"}), 400
        controller.set_mode("safe")
        with controller.lock:
            if (controller.goal_calibration_session and
                    controller.goal_calibration_color != color_name):
                controller._clear_goal_calibration(
                    "Goal color changed; press Start new calibration."
                )
            controller.selected_goal_color = color_name
            if not controller.goal_calibration_session:
                controller.goal_calibration_color = color_name
            controller.goal_target = None
        return jsonify(controller.snapshot())

    @app.post("/api/joystick")
    def joystick():
        data = request.get_json(force=True)
        controller.set_joystick(
            float(data["x"]), float(data["y"]), float(data.get("r", 0.0))
        )
        return jsonify({"ok": True})

    @app.post("/api/action")
    def action():
        controller.action(request.get_json(force=True)["action"])
        return jsonify({"ok": True})

    @app.post("/api/detection-lines")
    def detection_lines():
        data = request.get_json(force=True)
        try:
            controller.set_detection_lines(
                float(data["near_percent"]), float(data["capture_percent"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/goal-settings")
    def goal_settings():
        data = request.get_json(force=True)
        try:
            controller.set_goal_settings(
                float(data["shoot_percent"]), bool(data["auto_kick_enabled"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/auto-settings")
    def auto_settings():
        data = request.get_json(force=True)
        try:
            controller.set_auto_settings(
                int(data["dribbler_pwm"]),
                float(data["capture_push_time_s"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/course-settings")
    def course_settings():
        data = request.get_json(force=True)
        try:
            controller.set_course_settings(
                int(data["black_value_max"]),
                float(data["roi_top_percent"]),
                float(data["min_cone_area"]),
                float(data["forward_command"]),
                float(data["goal_stop_percent"]),
                float(data["side_danger_width_percent"]),
                float(data["orbit_start_percent"]),
                float(data["orbit_pass_distance_m"]),
                float(data["orbit_command"]),
                float(data["goal_search_progress_percent"]),
                float(data["post_clear_forward_s"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/field-map")
    def field_map():
        data = request.get_json(force=True)
        try:
            controller.set_field_map(
                float(data["width_m"]),
                float(data["length_m"]),
                float(data["odometry_x_scale"]),
                float(data["odometry_y_scale"]),
                float(data["waypoint_max_command"]),
                float(data["waypoint_arrival_radius_m"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/waypoint")
    def waypoint():
        data = request.get_json(force=True)
        try:
            controller.set_waypoint(float(data["x_m"]), float(data["y_m"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/waypoint/clear")
    def clear_waypoint():
        try:
            controller.clear_waypoint()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/odometry/reset")
    def reset_odometry():
        try:
            controller.reset_odometry()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/odometry/clear-trail")
    def clear_odometry_trail():
        controller.clear_odometry_trail()
        return jsonify(controller.snapshot())

    @app.post("/api/side-mask")
    def side_mask():
        data = request.get_json(force=True)
        try:
            controller.set_side_mask(data)
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/side-avoidance")
    def side_avoidance():
        data = request.get_json(force=True)
        try:
            controller.set_side_avoidance(bool(data["enabled"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/floor-calibration/begin")
    def begin_floor_calibration():
        try:
            controller.begin_floor_calibration()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/floor-calibration/sample")
    def sample_floor_calibration():
        data = request.get_json(force=True)
        try:
            controller.sample_floor_calibration(
                float(data["x"]), float(data["y"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/floor-calibration/apply")
    def apply_floor_calibration():
        try:
            controller.apply_floor_calibration()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/floor-calibration/cancel")
    def cancel_floor_calibration():
        controller.cancel_floor_calibration()
        return jsonify(controller.snapshot())

    @app.post("/api/ball-calibration/begin")
    def begin_ball_calibration():
        data = request.get_json(force=True)
        try:
            controller.begin_ball_calibration(str(data["color"]).lower())
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/ball-calibration/sample")
    def sample_ball_calibration():
        data = request.get_json(force=True)
        try:
            controller.sample_ball_calibration(float(data["x"]), float(data["y"]))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/ball-calibration/apply")
    def apply_ball_calibration():
        try:
            controller.apply_ball_calibration()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/ball-calibration/cancel")
    def cancel_ball_calibration():
        controller.cancel_ball_calibration()
        return jsonify(controller.snapshot())

    @app.post("/api/goal-calibration/begin")
    def begin_goal_calibration():
        data = request.get_json(force=True)
        try:
            controller.begin_goal_calibration(str(data["color"]).lower())
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/goal-calibration/sample")
    def sample_goal_calibration():
        data = request.get_json(force=True)
        try:
            controller.sample_goal_calibration(
                float(data["x1"]), float(data["y1"]),
                float(data["x2"]), float(data["y2"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/goal-calibration/apply")
    def apply_goal_calibration():
        try:
            controller.apply_goal_calibration()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(controller.snapshot())

    @app.post("/api/goal-calibration/cancel")
    def cancel_goal_calibration():
        controller.cancel_goal_calibration()
        return jsonify(controller.snapshot())

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--serial", default="COM3")
    parser.add_argument(
        "--side-camera", type=int, default=None,
        help="optional top/side-safety camera index, commonly 1",
    )
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    controller = SokaWebController(
        args.camera, None if args.no_serial else args.serial, args.side_camera
    )
    app = create_app(controller)
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
