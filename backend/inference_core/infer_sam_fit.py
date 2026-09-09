"""SAM mask-fit loop — interactive polygon fitting with image cleaning."""

from __future__ import annotations

import argparse
import sys
import termios
import tty
from pathlib import Path

from config import CONFIG_PATH, load_config
from util.fit_loop import run_fit_loop

_ROOT = Path(__file__).resolve().parent


def _wait_continue_or_esc() -> bool:
    print("Press Enter to continue, Esc to exit...", end=" ", flush=True)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                print()
                return False
            if ch in ("\r", "\n"):
                print()
                return True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SAM mask-fit loop with image cleaning")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--image",
        type=Path,
        default=_ROOT / "test_image" / "002.jpeg",
        help="Floor plan image path",
    )
    args = parser.parse_args(argv)

    run_fit_loop(
        load_config(args.config),
        args.image,
        wait_continue_or_esc=_wait_continue_or_esc,
    )


if __name__ == "__main__":
    main()
