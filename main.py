"""Real-time object detection and sprinkler control."""

from __future__ import annotations

import argparse
import logging
import time

import cv2
from ultralytics import YOLO

from config import load_config
from detection import Detection, detect_objects, draw_detections, should_activate
from sprinkler import SprinklerClient

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=None,
        help="Camera source: webcam index, video file, or RTSP URL. Defaults to .env.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Generic YOLO model path for standard classes (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--lawnmower-model",
        default=None,
        help="Optional custom YOLO model path for robot_lawnmower.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log sprinkler activations without sending HTTP requests.",
    )
    return parser.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    capture_source: int | str = int(source) if source.isdecimal() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {source}")
    return capture


def run(
    source: str | None = None,
    model_path: str = "yolov8n.pt",
    dry_run: bool = False,
    lawnmower_model_path: str | None = None,
) -> None:
    config = load_config()
    video_source = source or config.rtsp_url
    generic_model = YOLO(model_path)
    lawnmower_model = (
        YOLO(lawnmower_model_path) if lawnmower_model_path else None
    )
    capture = open_capture(video_source)
    sprinkler = SprinklerClient(
        config.api_url,
        config.sprinkler_time,
        dry_run=dry_run,
    )

    last_activation = float("-inf")

    logger.info("Started detection on %s", video_source)
    try:
        while True:
            success, frame = capture.read()
            if not success:
                logger.warning("Could not read a frame; stopping capture")
                break

            detections = detect_objects(
                generic_model.predict(frame, verbose=False, device="cpu"),
                generic_model.names,
                config.confidence_threshold,
            )
            if lawnmower_model is not None:
                detections.extend(
                    detect_objects(
                        lawnmower_model.predict(
                            frame,
                            verbose=False,
                            device="cpu",
                        ),
                        lawnmower_model.names,
                        config.confidence_threshold,
                    )
                )
            target_detected = any(
                detection.class_name in config.target_classes
                for detection in detections
            )
            now = time.monotonic()
            if should_activate(target_detected,now,last_activation,config.api_cooldown_seconds):
                sprinkler.activate()
                last_activation = now

            draw_detections(frame, detections)
            cv2.imshow("object detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Detection stopped")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    try:
        run(args.source, args.model, args.dry_run, args.lawnmower_model)
    except (KeyError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
