"""Field-relative single-waypoint motion planning for SOKA."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class WaypointDecision:
    x_command: float
    y_command: float
    distance_m: float
    arrived: bool


def plan_waypoint_motion(
    current_x_m: float,
    current_y_m: float,
    heading_deg: float,
    target_x_m: float,
    target_y_m: float,
    maximum_command: float,
    minimum_command: float,
    slow_radius_m: float,
    arrival_radius_m: float,
) -> WaypointDecision:
    """Return robot-relative X/Y commands for one field-relative target."""
    field_error_x = float(target_x_m) - float(current_x_m)
    field_error_y = float(target_y_m) - float(current_y_m)
    distance = math.hypot(field_error_x, field_error_y)
    if distance <= arrival_radius_m:
        return WaypointDecision(0.0, 0.0, distance, True)

    span = max(0.001, slow_radius_m - arrival_radius_m)
    progress = max(0.0, min(1.0, (distance - arrival_radius_m) / span))
    magnitude = minimum_command + (
        maximum_command - minimum_command
    ) * progress
    field_x = field_error_x / distance * magnitude
    field_y = field_error_y / distance * magnitude

    # Inverse of odometry's robot-to-field rotation. Heading zero points along
    # field +Y and positive heading is the verified clockwise/right turn.
    theta = math.radians(float(heading_deg))
    robot_x = field_x * math.cos(theta) - field_y * math.sin(theta)
    robot_y = field_x * math.sin(theta) + field_y * math.cos(theta)
    return WaypointDecision(robot_x, robot_y, distance, False)
