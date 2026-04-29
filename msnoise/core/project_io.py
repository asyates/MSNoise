"""Low-level I/O helpers for MSNoise project archives.

A *project archive* is a ``.tar.zst`` file containing a full MSNoise project
at a given pipeline level (preprocess / cc / stack / dvv …).  It is distinct
from an :class:`~msnoise.results.MSNoiseResult` *result bundle* (single
lineage, ``params.yaml`` + ``_output/``).

This module handles only raw filesystem operations (extract, list, checksum).
Higher-level logic lives in :mod:`msnoise.project`.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path


def extract_archive(src: str | Path, dest: str | Path) -> Path:
    """Extract a ``.tar.zst`` project archive into *dest*.

    The archive may contain files at the root or inside a single top-level
    directory — both layouts are handled transparently.  The extracted tree
    is always rooted at *dest* (no extra nesting introduced).

    :param src:  Path to the ``.tar.zst`` file.
    :param dest: Destination directory (created if absent).
    :returns:    Absolute :class:`~pathlib.Path` to the extracted project root
                 (i.e. the directory that contains ``project.yaml``).
    :raises FileNotFoundError: if *src* does not exist.
    :raises ValueError:        if *src* is not a recognised archive format.
    """
    import zstandard as zstd

    src = Path(src).resolve()
    dest = Path(dest).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Project archive not found: {src}")
    if not src.suffix == ".zst":
        raise ValueError(
            f"Expected a .tar.zst file, got: {src.name!r}. "
            "Use MSNoiseProject.from_project_dir() for an already-extracted project."
        )

    dest.mkdir(parents=True, exist_ok=True)

    with open(src, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tf:
                tf.extractall(dest)

    # Detect single top-level directory layout (common when archiving a folder)
    entries = [e for e in dest.iterdir() if not e.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and (entries[0] / "project.yaml").exists():
        return entries[0]

    # Otherwise project.yaml should be at dest root
    return dest


def file_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file (streaming, memory-efficient)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
