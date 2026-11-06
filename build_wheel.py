"""Build the wheel: the cubins of unica.PRESETS compiled fresh from csrc/ into unica/cubins/, then `uv build`.

usage: uv run python build_wheel.py        -> dist/unica_attention-<version>-py3-none-any.whl

One wheel for Windows and Linux: a cubin is the GPU's machine code, loaded by the driver on either. Their sha256 is
printed, to check that another machine loads the very same bytes. Needs nvcc; the GPU is not used.
"""

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

from unica import PRESETS, build_defines, cuda

ROOT = Path(__file__).resolve().parent
SOURCE = "flash"  # every preset's kernel and the quantizers live in csrc/flash.cu


def main() -> None:
    shutil.rmtree(cuda.SHIPPED, ignore_errors=True)
    cuda.SHIPPED.mkdir()
    # the quantizers run from the plain build, each preset's kernel from its own
    combinations = {cuda.cubin_name(SOURCE, d): d for d in [{}, *(build_defines(p) for p in PRESETS.values())]}
    for name, defines in combinations.items():
        (ROOT / "build" / name).unlink(missing_ok=True)  # cubin() checks only the .cu's time, not its headers
        (cuda.SHIPPED / name).write_bytes(cuda.cubin(SOURCE, defines))
        print(f"{name:40} {hashlib.sha256((cuda.SHIPPED / name).read_bytes()).hexdigest()}")
    subprocess.run(["uv", "build", "--wheel"], cwd=ROOT, check=True)
    wheel = max((ROOT / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
    shipped = {Path(n).name for n in zipfile.ZipFile(wheel).namelist() if n.endswith(".cubin")}
    if shipped != combinations.keys():
        raise SystemExit(f"{wheel.name} ships {sorted(shipped)}, expected {sorted(combinations)}")
    print(f"{wheel.relative_to(ROOT)}: {len(shipped)} cubins")


if __name__ == "__main__":
    main()
