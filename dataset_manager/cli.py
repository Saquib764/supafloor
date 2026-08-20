#!/usr/bin/env python3
"""Upload or download a registered dataset on Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from registry import download, list_datasets, load_dotenv, upload


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    uploadable = list_datasets(uploadable=True)

    parser = argparse.ArgumentParser(description="Dataset manager CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Push local data to Hugging Face")
    up.add_argument("dataset", choices=uploadable)
    up.add_argument("--revision", default=None)
    up.add_argument("--public", action="store_true")
    up.add_argument("--no-readme", action="store_true")

    down = sub.add_parser("download", help="Pull Hub data to local folders")
    down.add_argument("dataset", choices=uploadable)
    down.add_argument("--out", type=Path, default=None)
    down.add_argument("--revision", default=None)
    down.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "upload":
        upload(
            args.dataset,
            revision=args.revision,
            private=False if args.public else None,
            upload_readme=not args.no_readme,
        )
        return
    download(args.dataset, out=args.out, revision=args.revision, force=args.force)


if __name__ == "__main__":
    main()
