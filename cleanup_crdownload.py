"""Utility script to delete .jpg.crdownload files and report remaining .jpg files."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove leftover .jpg.crdownload files and show the count of remaining .jpg files"
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Directory to clean (defaults to the current working directory)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search for files recursively",
    )
    return parser.parse_args()


def collect_targets(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [
            path
            for path in root.rglob("*.jpg.crdownload")
            if path.is_file()
        ]
    return [
        path
        for path in root.iterdir()
        if path.is_file() and path.name.lower().endswith(".jpg.crdownload")
    ]


def count_jpg_files(root: Path, recursive: bool) -> int:
    if recursive:
        return sum(1 for path in root.rglob("*.jpg") if path.is_file())
    return sum(
        1
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    )


def main() -> None:
    args = parse_args()
    directory: Path = args.directory

    if not directory.exists():
        raise SystemExit(f"Directory '{directory}' does not exist.")
    if not directory.is_dir():
        raise SystemExit(f"'{directory}' is not a directory.")

    targets = collect_targets(directory, args.recursive)
    removed = 0
    for path in targets:
        path.unlink(missing_ok=True)
        removed += 1

    remaining_jpgs = count_jpg_files(directory, args.recursive)

    print(f"Removed {removed} .jpg.crdownload file(s).")
    print(f"Remaining .jpg files: {remaining_jpgs}")


if __name__ == "__main__":
    main()
