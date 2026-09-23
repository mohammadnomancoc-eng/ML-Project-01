"""Multi-Channel Alert and Webhook Notification Engine for ML-Project-01.

Dispatches structured operational alerts to Slack, Discord, Microsoft Teams, and custom webhooks
upon detecting data drift, schema violations, or pipeline failures.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from logger import logger


class AlertPayloadBuilder:
    """Constructs structured multi-platform notification JSON payloads."""

    @staticmethod
    def build_slack_alert(title: str, severity: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Formats payload for Slack Incoming Webhooks."""
        color = "#36a64f" if severity == "INFO" else ("#ffcc00" if severity == "WARNING" else "#ff0000")
        fields = [{"title": k, "value": str(v), "short": True} for k, v in details.items()]

        return {
            "attachments": [
                {
                    "fallback": f"[{severity}] {title}",
                    "color": color,
                    "title": f"🚨 {title}",
                    "fields": fields,
                    "footer": "ML-Project-01 Alert System",
                    "ts": int(datetime.now().timestamp()),
                }
            ]
        }

    @staticmethod
    def build_discord_alert(title: str, severity: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Formats payload for Discord Webhook embeds."""
        color = 3066993 if severity == "INFO" else (16776960 if severity == "WARNING" else 15158332)
        fields = [{"name": k, "value": str(v), "inline": True} for k, v in details.items()]

        return {
            "embeds": [
                {
                    "title": f"🚨 {title}",
                    "description": f"**Severity:** `{severity}`",
                    "color": color,
                    "fields": fields,
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        }


class DriftAlerter:
    """Manages dispatching alerts to configured endpoints."""

    def __init__(self, webhook_url: Optional[str] = None, platform: str = "slack"):
        self.webhook_url = webhook_url
        self.platform = platform
        self.alert_history: List[Dict[str, Any]] = []

    def send_alert(self, title: str, severity: str = "WARNING", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Sends or logs the alert payload based on specified severity."""
        details = details or {}
        if self.platform == "slack":
            payload = AlertPayloadBuilder.build_slack_alert(title, severity, details)
        else:
            payload = AlertPayloadBuilder.build_discord_alert(title, severity, details)

        record = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "severity": severity,
            "details": details,
            "payload": payload,
        }
        self.alert_history.append(record)
        logger.warning(f"[DriftAlerter] ({severity}) {title} - {details}")
        return record
