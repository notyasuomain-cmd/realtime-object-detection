"""Pure detection helpers and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    coordinates: tuple[int, int, int, int]


def detect_objects(
    results: Iterable[Any],
    names: dict[int, str],
    confidence_threshold: float,
) -> list[Detection]:
    detections: list[Detection] = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < confidence_threshold:
                continue
            class_name = names[int(box.cls[0])]
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
            detections.append(
                Detection(class_name, confidence, (x1, y1, x2, y2))
            )
    return detections


def should_activate(
    target_detected: bool,
    now: float,
    last_activation: float,
    cooldown_seconds: float,
) -> bool:
    return target_detected and now - last_activation >= cooldown_seconds


def draw_detections(frame: Any, detections: Iterable[Detection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{detection.class_name} {detection.confidence:.2f}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
