"""Utility for removing numbered duplicate files.

This script walks a directory tree and removes files whose names end with a
copy suffix such as ``" (1)"`` or ``"（2）"``.  It normalises unusual
whitespace and parenthesis characters before deciding whether a file is a
duplicate, so it also handles names produced by synchronisation tools like
Google Drive.

Example
-------
```bash
python remove_numbered_duplicates.py /path/to/dataset
```

The script keeps the first occurrence of each stem (per directory) and deletes
later copies.  If the first file encountered still contains the numbered suffix
it is renamed to the base name.
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple


# Match a trailing `` (digits)`` group after normalisation.  Some providers add
# extra whitespace inside the parentheses (e.g. ``" ( 1 )"``) so we accept
# optional spaces around the digits.
COPY_SUFFIX = re.compile(r"\s*\(\s*(\d+)\s*\)$")


def _normalise_stem(stem: str) -> str:
    """Return a canonical version of *stem* with copy suffix removed.

    The function performs the following clean-up steps:

    * Apply NFKC normalisation so that full-width parentheses are converted to
      the ASCII variants.
    * Replace any kind of Unicode whitespace with a plain ASCII space.
    * Remove zero-width "format" characters.
    * Collapse repeated whitespace into a single space.
    * Remove a trailing `` (digits)`` segment when present.
    """

    # First bring characters into a canonical compatibility form.
    normalised = unicodedata.normalize("NFKC", stem)

    cleaned_chars = []
    for char in normalised:
        category = unicodedata.category(char)
        if category == "Cf":
            # "Format" characters include zero-width joiners and friends; they
            # are invisible and can interfere with comparisons, so drop them.
            continue
        if category.startswith("Z"):
            # All separator characters (including non-breaking spaces) become a
            # regular space so that the regex below can handle them.
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(char)

    collapsed = re.sub(r"\s+", " ", "".join(cleaned_chars)).strip()

    while True:
        match = COPY_SUFFIX.search(collapsed)
        if not match:
            break
        # ``foo (1) (2)`` should normalise all the way down to ``foo``; loop so
        # that we strip every numbered suffix that appears.
        collapsed = collapsed[: match.start()].rstrip()

    return collapsed


def dedupe_directory(root: Path) -> Counter:
    """Remove duplicate files in *root* and return statistics."""

    stats: Counter = Counter()

    for current_root, _, files in os.walk(root):
        seen: Dict[Tuple[str, str], Path] = {}
        directory = Path(current_root)

        # Sorting keeps behaviour predictable and ensures consistent stats.
        for name in sorted(files):
            path = directory / name
            stem = path.stem
            suffix = path.suffix
            canonical_stem = _normalise_stem(stem)

            if not canonical_stem:
                # Skip files that would end up with an empty name after
                # canonicalisation; deleting them would likely be surprising.
                stats["skipped_empty"] += 1
                continue

            key = (canonical_stem.casefold(), suffix.casefold())
            target_name = f"{canonical_stem}{suffix.lower()}"
            target_path = directory / target_name

            if key in seen:
                # This is a duplicate of a file we've already kept.
                path.unlink()
                stats["deleted"] += 1
                continue

            # First time we've seen this stem + suffix.
            seen[key] = target_path

            if path == target_path:
                stats["kept"] += 1
                continue

            if target_path.exists():
                # Another file already lives at the canonical name.  To avoid
                # overwriting data we delete the numbered variant instead.
                path.unlink()
                stats["deleted"] += 1
                continue

            path.rename(target_path)
            stats["renamed"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Root of the dataset to clean",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        raise SystemExit(f"Directory {args.directory} does not exist")

    stats = dedupe_directory(args.directory)
    total_deleted = stats.get("deleted", 0)
    total_renamed = stats.get("renamed", 0)
    total_kept = stats.get("kept", 0)
    total_skipped = stats.get("skipped_empty", 0)

    print(
        "Cleanup complete. Deleted {deleted} duplicates, "
        "renamed {renamed}, kept {kept}, skipped {skipped}.".format(
            deleted=total_deleted,
            renamed=total_renamed,
            kept=total_kept,
            skipped=total_skipped,
        )
    )


if __name__ == "__main__":
    main()

