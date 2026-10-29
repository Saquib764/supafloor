"""Safe cache cleanup: runtime device memory and on-disk scratch only."""

from __future__ import annotations

import shutil
from pathlib import Path

from config import Config, project_root
from device import release_device_memory


def _remove_tree(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        size = path.stat().st_size
        path.unlink()
        return size
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    shutil.rmtree(path)
    return size


def clean_scratch(cfg: Config) -> list[str]:
    """Drop bytecode caches, stale hub locks, and HF xet scratch. Keeps model blobs."""
    root = project_root()
    freed = 0
    lines: list[str] = []

    for cache_dir in root.rglob("__pycache__"):
        if ".venv" in cache_dir.parts:
            continue
        nbytes = _remove_tree(cache_dir)
        freed += nbytes
        lines.append(f"removed {cache_dir.relative_to(root)} ({nbytes // 1024} KiB)")

    locks = cfg.hf_cache / "hub" / ".locks"
    if locks.is_dir():
        for lock in locks.iterdir():
            if lock.is_file():
                nbytes = _remove_tree(lock)
                freed += nbytes
                lines.append(f"removed hub lock {lock.name}")

    xet = cfg.hf_cache / "xet"
    if xet.is_dir():
        nbytes = _remove_tree(xet)
        freed += nbytes
        lines.append(f"removed {xet} ({nbytes // 1024} KiB)")

    if not lines:
        lines.append("nothing to remove (scratch already clean)")
    else:
        lines.append(f"scratch freed ~{freed // (1024 * 1024)} MiB")
    return lines


def run(cfg: Config, *, scratch: bool) -> None:
    release_device_memory()
    print("clean-cache: released device allocator cache in this process", flush=True)
    if scratch:
        for line in clean_scratch(cfg):
            print(f"clean-cache: {line}", flush=True)
