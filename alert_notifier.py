"""Webhook Notification and Alert Dispatcher."""

import json
import urllib.request
from typing import Dict, Any, Optional
from logger import get_logger

logger = get_logger("Notifier")


class AlertNotifier:
    """Sends webhook alerts (Slack, Discord, generic) for training runs & drift events."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url

    def send_alert(self, title: str, message: str, level: str = "info", details: Optional[Dict[str, Any]] = None) -> bool:
        """Dispatches structured JSON notification to configured webhook."""
        if not self.webhook_url:
            logger.info(f"[{level.upper()}] {title}: {message} (No webhook URL configured — logged to console)")
            return True

        payload = {
            "title": title,
            "message": message,
            "level": level,
            "details": details or {},
        }

        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status in (200, 204)
        except Exception as e:
            logger.warning(f"Failed to send webhook alert: {e}")
            return False
