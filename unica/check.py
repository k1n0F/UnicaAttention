"""Whether this machine can run unica's kernels. `python -m unica` prints it and runs each preset (unica/__main__.py).

The cubins are sm_120a, the GPU's own machine code for compute capability 12.0 (RTX 50xx, RTX PRO 6000 Blackwell):
another capability cannot load them at all. Within 12.0, only the combination below has been measured.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import torch

from unica import cuda

CAPABILITY = (12, 0)  # sm_120a is arch-specific: not even 12.1 runs it
MIN_DRIVER = 13000  # the cubins come from CUDA 13.2; a driver for CUDA 12 does not load them
# What the kernels have been measured on. Anything else that can load them runs, with a warning.
TESTED_GPU = "NVIDIA GeForce RTX 5080"
TESTED_DRIVER = 13040  # 617.14
TESTED_TORCH = "2.14"
TESTED_PLATFORMS = {"win32", "linux"}  # Linux: WSL2 on the same card (LOG M4.29); a native Linux driver is not yet

OK, UNTESTED, UNSUPPORTED = "ok", "untested", "unsupported"


class Compatibility(NamedTuple):
    status: str  # OK, UNTESTED (runs, but not measured on this combination) or UNSUPPORTED (cannot run)
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        head = {OK: "unica: tested on this machine", UNTESTED: "unica: runs here, but untested, may be unstable",
                UNSUPPORTED: "unica: cannot run here"}[self.status]
        return head + "".join(f"\n  - {reason}" for reason in self.reasons)


def machine() -> dict:
    """What compatibility() judges, read from this machine; None where there is no CUDA."""
    found = {"platform": sys.platform, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
             "gpu": None, "capability": None, "driver": None}
    if torch.cuda.is_available():
        found["gpu"] = torch.cuda.get_device_name()
        found["capability"] = torch.cuda.get_device_capability()
    try:
        found["driver"] = cuda.driver_version()
    except (OSError, RuntimeError):
        pass
    return found


def compatibility(found: dict | None = None) -> Compatibility:
    """OK, UNTESTED or UNSUPPORTED, with the reasons. found: what machine() returns, or other values to judge."""
    found = found or machine()
    cannot, untested = [], []
    if found["driver"] is None:
        cannot.append("no NVIDIA driver")
    elif found["driver"] < MIN_DRIVER:
        cannot.append(f"the driver supports CUDA {found['driver'] // 1000}.{found['driver'] % 1000 // 10}, "
                      f"the kernels need {MIN_DRIVER // 1000}.0 or newer")
    elif found["driver"] < TESTED_DRIVER:
        untested.append(f"driver for CUDA {found['driver'] // 1000}.{found['driver'] % 1000 // 10}, "
                        f"older than the tested {TESTED_DRIVER // 1000}.{TESTED_DRIVER % 1000 // 10}")
    if found["torch_cuda"] is None:
        cannot.append(f"torch {found['torch']} is built without CUDA")
    elif found["capability"] is None:
        cannot.append("torch sees no GPU")
    elif tuple(found["capability"]) != CAPABILITY:
        cannot.append(f"{found['gpu']} has compute capability {'.'.join(map(str, found['capability']))}; "
                      f"the kernels are built for {'.'.join(map(str, CAPABILITY))} only (RTX 50xx, RTX PRO Blackwell)")
    elif found["gpu"] != TESTED_GPU:
        untested.append(f"{found['gpu']}: same compute capability, measured only on the {TESTED_GPU}")
    if not found["torch"].startswith(TESTED_TORCH + "."):
        untested.append(f"torch {found['torch']}, tested with {TESTED_TORCH}")
    if found["platform"] not in TESTED_PLATFORMS:
        untested.append(f"platform {found['platform']}, tested on {', '.join(sorted(TESTED_PLATFORMS))}")
    status = UNSUPPORTED if cannot else UNTESTED if untested else OK
    return Compatibility(status, tuple(cannot or untested))
