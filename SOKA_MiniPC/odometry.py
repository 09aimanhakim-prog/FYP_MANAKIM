"""Encoder and gyro based field-relative odometry for the SOKA robot."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


WHEEL_DIAMETER_M = 0.060
WHEEL_CIRCUMFERENCE_M = math.pi * WHEEL_DIAMETER_M
MAX_PLAUSIBLE_DRIVE_RPM = 650.0
MAX_INTEGRATION_DT_S = 0.35
TRAIL_MIN_DISTANCE_M = 0.025
TRAIL_MAX_POINTS = 260


@dataclass
class OdometrySnapshot:
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    speed_mps: float
    heading_deg: float
    rpm1: float
    rpm2: float
    rpm3: float
    telemetry_age_s: float | None
    valid: bool
    trail: list[tuple[float, float]]


class EncoderOdometry:
    """Integrate three omni-wheel RPM values into a relative field pose."""

    def __init__(self) -> None:
        self.x_m = 0.0
        self.y_m = 0.0
        self.vx_mps = 0.0
        self.vy_mps = 0.0
        self.heading_deg = 0.0
        self.rpm1 = self.rpm2 = self.rpm3 = 0.0
        self.last_esp_ms: int | None = None
        self.last_pc_time: float | None = None
        self.valid = False
        self.trail: list[tuple[float, float]] = [(0.0, 0.0)]

    def reset(self, x_m: float = 0.0, y_m: float = 0.0) -> None:
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.vx_mps = 0.0
        self.vy_mps = 0.0
        self.last_esp_ms = None
        self.last_pc_time = None
        self.valid = False
        self.trail = [(self.x_m, self.y_m)]

    def clear_trail(self) -> None:
        self.trail = [(self.x_m, self.y_m)]

    def update(self, esp_ms: int, rpm1: float, rpm2: float, rpm3: float,
               heading_deg: float, x_scale: float, y_scale: float) -> bool:
        now = time.monotonic()
        rpms = (float(rpm1), float(rpm2), float(rpm3))
        self.rpm1, self.rpm2, self.rpm3 = rpms
        self.heading_deg = float(heading_deg)
        self.last_pc_time = now

        if any(not math.isfinite(value) for value in rpms):
            self.valid = False
            return False
        if any(abs(value) > MAX_PLAUSIBLE_DRIVE_RPM for value in rpms):
            self.vx_mps = self.vy_mps = 0.0
            self.valid = False
            self.last_esp_ms = int(esp_ms) & 0xFFFFFFFF
            return False

        current_ms = int(esp_ms) & 0xFFFFFFFF
        if self.last_esp_ms is None:
            self.last_esp_ms = current_ms
            self.valid = True
            return False
        elapsed_ms = (current_ms - self.last_esp_ms) & 0xFFFFFFFF
        self.last_esp_ms = current_ms
        dt = elapsed_ms * 0.001
        if dt <= 0.0 or dt > MAX_INTEGRATION_DT_S:
            self.vx_mps = self.vy_mps = 0.0
            self.valid = False
            return False

        v1, v2, v3 = (
            value * WHEEL_CIRCUMFERENCE_M / 60.0 for value in rpms
        )
        # Least-squares inverse of the tested drive equations:
        # m1=.5X+.866Y, m2=.5X-.866Y, m3=-X. Common wheel rotation
        # is orthogonal to X/Y and therefore cancels here.
        robot_vx = ((v1 + v2 - 2.0 * v3) / 3.0) * float(x_scale)
        robot_vy = (0.577350269 * (v1 - v2)) * float(y_scale)

        theta = math.radians(self.heading_deg)
        # Heading zero points along field +Y. On the physically verified robot,
        # positive heading is a clockwise/right turn.
        field_vx = robot_vx * math.cos(theta) + robot_vy * math.sin(theta)
        field_vy = -robot_vx * math.sin(theta) + robot_vy * math.cos(theta)
        self.x_m += field_vx * dt
        self.y_m += field_vy * dt
        self.vx_mps = field_vx
        self.vy_mps = field_vy
        self.valid = True

        last_x, last_y = self.trail[-1]
        if math.hypot(self.x_m - last_x, self.y_m - last_y) >= TRAIL_MIN_DISTANCE_M:
            self.trail.append((self.x_m, self.y_m))
            if len(self.trail) > TRAIL_MAX_POINTS:
                self.trail = self.trail[-TRAIL_MAX_POINTS:]
        return True

    def snapshot(self) -> OdometrySnapshot:
        age = (
            None if self.last_pc_time is None
            else max(0.0, time.monotonic() - self.last_pc_time)
        )
        stale = age is None or age > 0.6
        vx = 0.0 if stale else self.vx_mps
        vy = 0.0 if stale else self.vy_mps
        return OdometrySnapshot(
            self.x_m,
            self.y_m,
            vx,
            vy,
            math.hypot(vx, vy),
            self.heading_deg,
            self.rpm1,
            self.rpm2,
            self.rpm3,
            age,
            self.valid and not stale,
            list(self.trail),
        )
