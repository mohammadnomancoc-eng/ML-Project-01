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

    @staticmethod
    def check_disk_space(path: str = ".") -> Dict[str, Any]:
        """Checks available disk storage in GB and percentage free."""
        import shutil

        total, used, free = shutil.disk_usage(path)
        return {
            "total_gb": round(total / (2**30), 2),
            "used_gb": round(used / (2**30), 2),
            "free_gb": round(free / (2**30), 2),
            "percent_free": round((free / total) * 100, 2),
            "healthy": (free / total) > 0.05,
        }

    @staticmethod
    def is_healthy(min_free_disk_percent: float = 5.0) -> bool:
        """Determines if host system meets minimum runtime thresholds."""
        disk = SystemHealthMonitor.check_disk_space()
        return disk["percent_free"] >= min_free_disk_percent

    @staticmethod
    def check_hardware_acceleration() -> Dict[str, Any]:
        """Detects whether hardware acceleration or CUDA drivers are available."""
        cuda_available = False
        device_name = "CPU"

        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        return {
            "cuda_available": cuda_available,
            "primary_device": device_name,
        }

