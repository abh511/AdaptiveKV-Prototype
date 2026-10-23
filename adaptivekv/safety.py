"""
adaptivekv/safety.py
--------------------
In-process hardware guard for long experiments on a laptop GPU.

guard() is called by the experiment scripts before every model run:
  * GPU temperature >= pause_temp_c  -> sleep until it is <= resume_temp_c
    (no progress is lost; the pause is logged)
  * free system RAM < min_free_ram_mb -> raise SafetyStop (clean exit)

It lives inside the experiment process on purpose: an external watchdog can be
killed separately and leave the experiment running unguarded.
Thresholds: `safety` block in config.yaml.
"""

import subprocess
import time


class SafetyStop(RuntimeError):
    pass


def gpu_temp_c() -> int:
    """GPU temperature via nvidia-smi; -1 if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def free_ram_mb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


class Guard:
    def __init__(self, cfg: dict):
        s = cfg.get("safety", {})
        self.pause_c = s.get("pause_temp_c", 80)
        self.resume_c = s.get("resume_temp_c", 70)
        self.min_ram = s.get("min_free_ram_mb", 1500)
        self.max_seen = 0
        self.pauses = 0
        self.paused_s = 0.0

    def __call__(self) -> None:
        ram = free_ram_mb()
        if 0 <= ram < self.min_ram:
            raise SafetyStop(f"free RAM {ram} MB < {self.min_ram} MB")
        t = gpu_temp_c()
        self.max_seen = max(self.max_seen, t)
        if t >= self.pause_c:
            self.pauses += 1
            t0 = time.time()
            print(f"  [safety] GPU {t} C >= {self.pause_c} C, pausing until <= {self.resume_c} C", flush=True)
            while t > self.resume_c:
                time.sleep(5)
                t = gpu_temp_c()
                if t < 0:
                    break
            self.paused_s += time.time() - t0
            print(f"  [safety] resumed at {t} C after {time.time() - t0:.0f} s", flush=True)

    def summary(self) -> str:
        return (f"max GPU temp {self.max_seen} C, {self.pauses} thermal pauses "
                f"({self.paused_s:.0f} s total; pause at {self.pause_c} C, resume at {self.resume_c} C)")
