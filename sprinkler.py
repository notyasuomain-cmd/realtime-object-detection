"""HTTP client for the sprinkler controller."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


class SprinklerClient:
    def __init__(self, api_url: str | None, seconds: int, dry_run: bool = False) -> None:
        self.api_url = api_url
        self.seconds = seconds
        self.dry_run = dry_run

    def activate(self) -> None:
        if self.dry_run:
            logger.info("[DRY RUN] Sprinkler would activate for %s seconds", self.seconds)
            return
        if not self.api_url:
            logger.error("WEB_SERVER_URL is not configured; sprinkler not activated")
            return

        try:
            response = requests.post(
                self.api_url,
                json={"seconds": self.seconds},
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception("Sprinkler API request failed")
            return
        logger.info("Sprinkler activated for %s seconds", self.seconds)
