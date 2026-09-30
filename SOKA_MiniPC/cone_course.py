"""Front-camera dark floor-obstacle detection and local planning for SOKA."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class ConeDetection:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    fill_ratio: float
    track_id: int = -1
    confirmations: int = 1

    @property
    def bottom(self) -> int:
        return self.bbox[1] + self.bbox[3]


@dataclass
class PartialGoal:
    center_x: float
    bottom: int
    area: float


@dataclass
class CourseDecision:
    target_x: float
    x_command: float
    y_command: float
    status: str
    nearest_bottom_ratio: float
    path_blocked: bool
    imminent_threat: bool
    escape_direction: float
    forward_corridor_blocked: bool


def detect_black_cones(
    hsv: np.ndarray,
    value_max: int,
    roi_top_ratio: float,
    min_area: float,
) -> tuple[list[ConeDetection], np.ndarray]:
    """Return dark floor obstacles, including cones and irregular objects."""
    height, width = hsv.shape[:2]
    roi_top = max(0, min(height - 1, round(height * roi_top_ratio)))
    mask = cv2.inRange(
        hsv,
        np.array((0, 0, 0), dtype=np.uint8),
        np.array((179, 255, int(value_max)), dtype=np.uint8),
    )
    mask[:roi_top, :] = 0
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections: list[ConeDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 7 or box_height < 14:
            continue
        # Accept wider objects than the original cone-only detector. Very thin
        # horizontal regions remain excluded because they are usually seams,
        # field markings, or shadows rather than standing obstacles.
        aspect = box_width / float(box_height)
        if not 0.10 <= aspect <= 6.0:
            continue
        fill = area / float(max(1, box_width * box_height))
        if fill < 0.14:
            continue
        if box_width > width * 0.88 or box_height > height * 0.88:
            continue
        detections.append(
            ConeDetection(
                (x, y, box_width, box_height),
                (x + box_width // 2, y + box_height // 2),
                area,
                fill,
            )
        )

    detections.sort(key=lambda item: item.bottom, reverse=True)
    return detections, clean


class ObstacleTracker:
    """Small nearest-neighbour tracker that rejects one-frame dark noise."""

    def __init__(self, confirm_frames: int = 3, max_missed_frames: int = 5,
                 smoothing: float = 0.45) -> None:
        self.confirm_frames = confirm_frames
        self.max_missed_frames = max_missed_frames
        self.smoothing = smoothing
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: list[ConeDetection]) -> list[ConeDetection]:
        unmatched = set(range(len(detections)))
        for track_id, track in list(self._tracks.items()):
            old: ConeDetection = track["detection"]
            best_index = None
            best_cost = float("inf")
            for index in unmatched:
                candidate = detections[index]
                dx = candidate.center[0] - old.center[0]
                db = candidate.bottom - old.bottom
                max_dx = max(55.0, 0.75 * max(old.bbox[2], candidate.bbox[2]))
                if abs(dx) > max_dx or abs(db) > 110.0:
                    continue
                cost = abs(dx) + 0.65 * abs(db)
                if cost < best_cost:
                    best_cost = cost
                    best_index = index
            if best_index is None:
                track["misses"] += 1
                if track["misses"] > self.max_missed_frames:
                    del self._tracks[track_id]
                continue

            candidate = detections[best_index]
            unmatched.remove(best_index)
            alpha = self.smoothing
            old_box = old.bbox
            new_box = candidate.bbox
            box = tuple(round((1.0 - alpha) * a + alpha * b)
                        for a, b in zip(old_box, new_box))
            hits = track["hits"] + 1
            smoothed = ConeDetection(
                box,
                (box[0] + box[2] // 2, box[1] + box[3] // 2),
                (1.0 - alpha) * old.area + alpha * candidate.area,
                (1.0 - alpha) * old.fill_ratio + alpha * candidate.fill_ratio,
                track_id,
                hits,
            )
            track.update(detection=smoothed, hits=hits, misses=0)

        for index in unmatched:
            detection = detections[index]
            track_id = self._next_id
            self._next_id += 1
            tracked = ConeDetection(
                detection.bbox, detection.center, detection.area,
                detection.fill_ratio, track_id, 1,
            )
            self._tracks[track_id] = {
                "detection": tracked, "hits": 1, "misses": 0,
            }

        result = [
            track["detection"]
            for track in self._tracks.values()
            if track["hits"] >= self.confirm_frames
            and track["misses"] <= self.max_missed_frames
        ]
        result.sort(key=lambda item: item.bottom, reverse=True)
        return result


def partial_goal_from_mask(mask: np.ndarray, min_area: float = 350.0) -> PartialGoal | None:
    """Estimate a colored goal span even when black cones split it into pieces."""
    height, width = mask.shape[:2]
    clean = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)),
        iterations=1,
    )
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    pieces: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 12 or box_height < 10 or y > height * 0.88:
            continue
        pieces.append((area, (x, y, box_width, box_height)))
    if not pieces:
        return None

    # Merge up to three large colored fragments. Cone occlusion commonly turns
    # one goal interior into a left and right component.
    pieces.sort(key=lambda item: item[0], reverse=True)
    primary_area, primary_box = pieces[0]
    px, py, pw, ph = primary_box
    primary_mid_y = py + ph * 0.5
    accepted = [(primary_area, primary_box)]
    for area, box in pieces[1:3]:
        x, y, box_width, box_height = box
        mid_y = y + box_height * 0.5
        if abs(mid_y - primary_mid_y) <= max(ph, box_height) * 0.85:
            accepted.append((area, box))

    left = min(box[0] for _, box in accepted)
    right = max(box[0] + box[2] for _, box in accepted)
    bottom = max(box[1] + box[3] for _, box in accepted)
    total_area = sum(area for area, _ in accepted)
    center_x = max(0.0, min(float(width - 1), (left + right) * 0.5))
    return PartialGoal(center_x, bottom, total_area)


def plan_course_motion(
    frame_width: int,
    frame_height: int,
    cones: list[ConeDetection],
    preferred_x: float,
    previous_target_x: float,
    clear_forward_command: float,
    danger_bottom_ratio: float = 0.72,
    side_danger_width_ratio: float = 0.34,
) -> CourseDecision:
    """Choose a smooth free lateral corridor and X/Y vector command."""
    camera_center = frame_width * 0.5
    candidates = [frame_width * value for value in (0.18, 0.29, 0.40, 0.50, 0.60, 0.71, 0.82)]
    preferred_x = max(frame_width * 0.12, min(frame_width * 0.88, preferred_x))
    previous_target_x = max(
        frame_width * 0.12, min(frame_width * 0.88, previous_target_x)
    )

    nearest = max((item.bottom / frame_height for item in cones), default=0.0)
    imminent_threat = False
    escape_direction = 0.0
    left_threat_bottom = 0.0
    right_threat_bottom = 0.0
    # The setting is the half-width of the robot's projected forward corridor,
    # measured from image center. Objects wholly outside this corridor are
    # tracked and drawn but do not force an unnecessary escape.
    corridor_left = frame_width * (0.5 - side_danger_width_ratio)
    corridor_right = frame_width * (0.5 + side_danger_width_ratio)
    forward_corridor_blocked = False
    for cone in cones:
        x, _y, box_width, _box_height = cone.bbox
        bottom_ratio = cone.bottom / frame_height
        proximity = max(0.0, min(1.0, (bottom_ratio - 0.32) / 0.68))
        safety_padding = 14.0 + 55.0 * proximity
        inflated_left = x - safety_padding
        inflated_right = x + box_width + safety_padding
        central_danger = (
            inflated_left < corridor_right
            and inflated_right > corridor_left
        )
        forward_corridor_blocked = forward_corridor_blocked or central_danger
        if bottom_ratio < danger_bottom_ratio:
            continue
        if not central_danger:
            continue
        imminent_threat = True
        if cone.center[0] < camera_center:
            left_threat_bottom = max(left_threat_bottom, bottom_ratio)
        else:
            right_threat_bottom = max(right_threat_bottom, bottom_ratio)

    if left_threat_bottom and right_threat_bottom:
        # Escape away from the closer/deeper side. If both sides are almost
        # equally blocked, direction zero tells the caller to stop instead of
        # blindly strafing into another cone.
        if abs(left_threat_bottom - right_threat_bottom) >= 0.04:
            escape_direction = (
                1.0 if left_threat_bottom > right_threat_bottom else -1.0
            )
    elif left_threat_bottom:
        escape_direction = 1.0
    elif right_threat_bottom:
        escape_direction = -1.0
    scored: list[tuple[float, float, bool]] = []
    for candidate_x in candidates:
        cost = abs(candidate_x - preferred_x) + 0.48 * abs(
            candidate_x - previous_target_x
        )
        blocked_close = False
        for cone in cones:
            x, _y, box_width, _box_height = cone.bbox
            bottom_ratio = cone.bottom / frame_height
            proximity = max(0.0, min(1.0, (bottom_ratio - 0.32) / 0.68))
            # Perspective-aware robot-radius inflation in image pixels.
            clearance = box_width * 0.5 + 42.0 + 135.0 * proximity
            separation = abs(candidate_x - cone.center[0])
            if separation < clearance:
                cost += (clearance - separation) * (5.0 + 18.0 * proximity)
                if bottom_ratio >= 0.84:
                    blocked_close = True
        scored.append((cost, candidate_x, blocked_close))

    _cost, target_x, path_blocked = min(scored, key=lambda item: item[0])
    error = target_x - camera_center
    if abs(error) <= 14.0:
        x_command = 0.0
    else:
        x_command = math.copysign(
            max(24.0, min(72.0, abs(error) / (frame_width * 0.5) * 82.0)),
            error,
        )

    forward = float(clear_forward_command)
    if nearest >= danger_bottom_ratio and imminent_threat:
        # Once a cone reaches the robot's immediate forward corridor, lateral
        # clearance takes priority. Advancing while strafing was making the
        # robot cut diagonally toward very close cones.
        forward = 0.0
    elif nearest >= 0.90:
        forward = 0.0 if path_blocked else min(forward, 16.0)
    elif nearest >= 0.80:
        forward = min(forward, 28.0)
    elif nearest >= 0.68:
        forward = min(forward, 40.0)

    if imminent_threat and nearest >= danger_bottom_ratio:
        status = (
            "COURSE CLOSE CONE - STRAFE ONLY"
            if x_command else "COURSE CLOSE CONE - STOP"
        )
    elif path_blocked and nearest >= 0.94:
        status = "COURSE BLOCKED - STRAFING" if x_command else "COURSE BLOCKED - STOP"
    elif cones:
        status = "COURSE AVOIDING CONES"
    else:
        status = "COURSE CLEAR - ADVANCING"
    return CourseDecision(
        target_x, x_command, forward, status, nearest, path_blocked,
        imminent_threat, escape_direction, forward_corridor_blocked,
    )
