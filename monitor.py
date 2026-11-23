"""Best-effort monitoring of system-level CPU, RAM and NVIDIA GPU load.

System-wide metrics are *not* process-specific. They may include unrelated activity.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass
from statistics import mean

import psutil


def _gpu_query(query: str) -> list[list[str]]:
    """Read NVIDIA-SMI CSV rows, returning [] if NVIDIA-SMI is not available."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=2,
        ).stdout
        return [[item.strip() for item in line.split(",")] for line in output.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def gpu_snapshot() -> dict[str, float] | None:
    data = _gpu_query("utilization.gpu,memory.used,memory.total")
    try:
        if not data:
            return None
        values = [[float(v) for v in row] for row in data]
        return {"utilization_pct": max(r[0] for r in values),
                "memory_used_mb": sum(r[1] for r in values),
                "memory_total_mb": sum(r[2] for r in values)}
    except (ValueError, IndexError):
        return None


def hardware_info() -> dict[str, object]:
    gpu = _gpu_query("name,memory.total")
    return {
        "os": platform.platform(),
        "processor": platform.processor() or "Non communiqué par le système",
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "gpu": [{"name": r[0], "vram_total_mb": r[1]} for r in gpu if len(r) >= 2],
    }


@dataclass
class ResourceSummary:
    samples: int = 0
    cpu_avg_pct: float | None = None
    cpu_peak_pct: float | None = None
    ram_peak_pct: float | None = None
    ram_peak_gb: float | None = None
    gpu_avg_pct: float | None = None
    gpu_peak_pct: float | None = None
    vram_peak_mb: float | None = None

    def as_dict(self) -> dict:
        return vars(self).copy()


class ResourceSampler:
    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._readings: list[dict] = []

    def _sample(self) -> None:
        vm = psutil.virtual_memory()
        reading: dict = {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_pct": vm.percent,
            "ram_gb": vm.used / 1024**3,
            "gpu": gpu_snapshot(),
        }
        self._readings.append(reading)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass  # Monitoring must not break generation
            self._stop.wait(self.interval)

    def start(self) -> None:
        psutil.cpu_percent(interval=None)  # Prime CPU delta meter.
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> ResourceSummary:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        rows = self._readings
        if not rows:
            return ResourceSummary()
        gpu_rows = [r["gpu"] for r in rows if r["gpu"] is not None]
        return ResourceSummary(
            samples=len(rows),
            cpu_avg_pct=round(mean(r["cpu_pct"] for r in rows), 2),
            cpu_peak_pct=round(max(r["cpu_pct"] for r in rows), 2),
            ram_peak_pct=round(max(r["ram_pct"] for r in rows), 2),
            ram_peak_gb=round(max(r["ram_gb"] for r in rows), 3),
            gpu_avg_pct=round(mean(r["utilization_pct"] for r in gpu_rows), 2) if gpu_rows else None,
            gpu_peak_pct=round(max(r["utilization_pct"] for r in gpu_rows), 2) if gpu_rows else None,
            vram_peak_mb=round(max(r["memory_used_mb"] for r in gpu_rows), 1) if gpu_rows else None,
        )
