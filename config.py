"""Application configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class AppConfig:
    rtsp_url: str
    api_url: str | None
    sprinkler_time: int
    api_cooldown_seconds: float
    confidence_threshold: float
    target_classes: frozenset[str]


def _positive_number(values: dict[str, str | None], name: str, default: str) -> float:
    raw_value = values.get(name, default)
    try:
        value = float(raw_value or default)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def load_config(env_path: Path = Path(".env")) -> AppConfig:
    values = dotenv_values(env_path)
    required = ("RTSP_USER", "RTSP_PASSWD", "CAM_IP", "RTSP_PORT")
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise KeyError(f"Missing configuration: {', '.join(missing)}")

    port = values["RTSP_PORT"]
    assert port is not None
    rtsp_url = (
        f"rtsp://{values['RTSP_USER']}:{values['RTSP_PASSWD']}"
        f"@{values['CAM_IP']}:{port}/Streaming/channels/101"
    )
    sprinkler_time = int(_positive_number(values, "SPRINKLER_TIME", "5"))
    confidence_threshold = _positive_number(values, "CONFIDENCE_THRESHOLD", "0.75")
    if confidence_threshold > 1:
        raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1")

    target_classes = frozenset(
        name.strip()
        for name in (values.get("TARGET_CLASSES") or "cat").split(",")
        if name.strip()
    )
    if not target_classes:
        raise ValueError("TARGET_CLASSES must contain at least one class")

    return AppConfig(
        rtsp_url=rtsp_url,
        api_url=values.get("WEB_SERVER_URL") or None,
        sprinkler_time=sprinkler_time,
        api_cooldown_seconds=_positive_number(
            values, "CAT_API_COOLDOWN_SECONDS", "60"
        ),
        confidence_threshold=confidence_threshold,
        target_classes=target_classes,
    )
