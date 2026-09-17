"""System Resource Monitoring and Training Node Telemetry."""

import os
import platform
from typing import Dict, Any
from logger import get_logger

logger = get_logger("SystemHealth")


class SystemHealthMonitor:
    """Profiles operating system memory, core counts, and platform hardware telemetry."""

    @staticmethod
    def get_system_snapshot() -> Dict[str, Any]:
        """Captures host system environmental metrics and hardware details."""
        cpu_count = os.cpu_count() or 1
        system_info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_cores": cpu_count,
            "pid": os.getpid(),
        }

        logger.info(
            f"System Snapshot: {system_info['os']} {system_info['os_release']} | {cpu_count} CPU cores | Python {system_info['python_version']}"
        )
        return system_info
