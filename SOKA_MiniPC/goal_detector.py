"""HSV and geometry based indoor soccer-goal detection for SOKA."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ball_tracker import hue_mask


CONFIG_PATH = Path(__file__).with_name("goal_colors.json")


@dataclass
class GoalFilter:
    h_center: int
    h_tolerance: int
    s_min: int
    v_min: int
    min_area: int
    min_aspect_ratio: float
    max_aspect_ratio: float
    min_rectangularity: float


@dataclass
class GoalDetection:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    area: float
    aspect_ratio: float
    rectangularity: float
    edge_support: float
    confidence: float
    contour: np.ndarray

    @property
    def bottom(self) -> int:
        return self.bbox[1] + self.bbox[3]


DEFAULTS = {
    "yellow": GoalFilter(30, 15, 80, 60, 1400, 1.10, 6.0, 0.28),
    "blue": GoalFilter(110, 18, 75, 50, 1400, 1.10, 6.0, 0.28),
}


def load_goal_filters() -> dict[str, GoalFilter]:
    if not CONFIG_PATH.exists():
        return {name: GoalFilter(**asdict(value)) for name, value in DEFAULTS.items()}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return {name: GoalFilter(**data[name]) for name in DEFAULTS}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Warning: could not load {CONFIG_PATH.name}: {exc}; using defaults")
        return {name: GoalFilter(**asdict(value)) for name, value in DEFAULTS.items()}


def save_goal_filters(filters: dict[str, GoalFilter]) -> None:
    CONFIG_PATH.write_text(
        json.dumps({name: asdict(value) for name, value in filters.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved goal color settings to {CONFIG_PATH}")


def find_goal_candidates(
    hsv: np.ndarray, settings: GoalFilter
) -> tuple[list[GoalDetection], np.ndarray]:
    mask = hue_mask(hsv, settings)
    kernel = np.ones((7, 7), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(clean.shape[0] * clean.shape[1])
    detections: list[GoalDetection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < settings.min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 35 or height < 18:
            continue
        aspect = width / float(height)
        if not settings.min_aspect_ratio <= aspect <= settings.max_aspect_ratio:
            continue
        rectangularity = area / float(width * height)
        if rectangularity < settings.min_rectangularity:
            continue

        # The colored goal interior should provide both side boundaries and a
        # lower horizontal boundary. Filled rectangular interiors also score
        # highly, while small round balls and thin random streaks do not.
        strip_x = max(2, round(width * 0.10))
        strip_y = max(2, round(height * 0.12))
        roi = clean[y:y + height, x:x + width]
        left = float(np.mean(roi[:, :strip_x]) / 255.0)
        right = float(np.mean(roi[:, width - strip_x:]) / 255.0)
        bottom = float(np.mean(roi[height - strip_y:, :]) / 255.0)
        edge_support = (left + right + bottom) / 3.0
        if edge_support < 0.16:
            continue

        aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 3.8)
        area_score = min(1.0, area / max(1.0, frame_area * 0.10))
        confidence = 100.0 * (
            0.35 * min(1.0, rectangularity)
            + 0.30 * min(1.0, edge_support)
            + 0.20 * aspect_score
            + 0.15 * area_score
        )
        detections.append(
            GoalDetection(
                (x, y, width, height),
                (x + width // 2, y + height // 2),
                area,
                aspect,
                rectangularity,
                edge_support,
                confidence,
                contour,
            )
        )

    return detections, clean


def select_locked_goal(
    candidates: list[GoalDetection], previous: GoalDetection | None
) -> GoalDetection | None:
    if not candidates:
        return None
    if previous is None:
        return max(candidates, key=lambda item: item.confidence + math.sqrt(item.area))

    px, py = previous.center

    def score(item: GoalDetection) -> float:
        dx = item.center[0] - px
        dy = item.center[1] - py
        distance_penalty = math.hypot(dx, dy) * 0.12
        return item.confidence + math.sqrt(item.area) - distance_penalty

    return max(candidates, key=score)
