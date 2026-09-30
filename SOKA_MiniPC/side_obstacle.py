"""Calibrated floor-color rejection for the SOKA side safety camera."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


CONFIG_PATH = Path(__file__).with_name("floor_color.json")


@dataclass
class FloorModel:
    l_center: int
    a_center: int
    b_center: int
    l_tolerance: int
    a_tolerance: int
    b_tolerance: int
    min_blob_area: int = 350


@dataclass
class ZoneObstacle:
    detected: bool
    area: float
    area_ratio: float
    bbox: tuple[int, int, int, int] | None


def load_floor_model() -> FloorModel | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        return FloorModel(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as exc:
        print(f"Warning: could not load {CONFIG_PATH.name}: {exc}")
        return None


def save_floor_model(model: FloorModel) -> None:
    CONFIG_PATH.write_text(
        json.dumps(asdict(model), indent=2), encoding="utf-8"
    )
    print(f"Saved floor color settings to {CONFIG_PATH}")


def fit_floor_model(lab_pixels: np.ndarray) -> tuple[FloorModel, str]:
    if len(lab_pixels) < 250:
        raise ValueError("not enough floor pixels; capture more floor samples")
    pixels = lab_pixels.astype(np.float64)
    center = np.median(pixels, axis=0)
    deviation = np.abs(pixels - center)
    spread = np.percentile(deviation, 95, axis=0)

    # L is allowed to vary more for lighting and soft shadows. The a/b chroma
    # channels remain tighter and carry most of the floor-color identity.
    tolerance = np.array(
        [
            np.clip(spread[0] + 14.0, 18.0, 65.0),
            np.clip(spread[1] + 7.0, 8.0, 38.0),
            np.clip(spread[2] + 7.0, 8.0, 38.0),
        ]
    )
    model = FloorModel(
        *(int(round(value)) for value in center),
        *(int(round(value)) for value in tolerance),
    )
    summary = (
        f"LAB=({model.l_center},{model.a_center},{model.b_center}) "
        f"tol=({model.l_tolerance},{model.a_tolerance},{model.b_tolerance})"
    )
    return model, summary


def foreign_pixel_mask(lab: np.ndarray, model: FloorModel) -> np.ndarray:
    center = np.array(
        [model.l_center, model.a_center, model.b_center], dtype=np.int16
    )
    tolerance = np.array(
        [model.l_tolerance, model.a_tolerance, model.b_tolerance], dtype=np.int16
    )
    difference = np.abs(lab.astype(np.int16) - center)
    floor = np.all(difference <= tolerance, axis=2)
    return np.where(floor, 0, 255).astype(np.uint8)


def analyze_zone(
    foreign: np.ndarray,
    rectangle: tuple[int, int, int, int],
    model: FloorModel,
) -> tuple[ZoneObstacle, np.ndarray]:
    left, top, right, bottom = rectangle
    zone = foreign[top:bottom, left:right]
    if zone.size == 0:
        return ZoneObstacle(False, 0.0, 0.0, None), np.zeros_like(foreign)

    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(zone, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum = max(model.min_blob_area, round(zone.size * 0.012))
    best_area = 0.0
    best_bbox = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area > best_area:
            x, y, width, height = cv2.boundingRect(contour)
            best_area = area
            best_bbox = (left + x, top + y, width, height)

    detected = best_area >= minimum
    area_ratio = best_area / float(zone.size)
    full_clean = np.zeros_like(foreign)
    full_clean[top:bottom, left:right] = clean
    return ZoneObstacle(detected, best_area, area_ratio, best_bbox), full_clean
