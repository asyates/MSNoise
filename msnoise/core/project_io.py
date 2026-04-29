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
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Keys in project.yaml step dicts that are workflow metadata, not config values.
_STEP_META_KEYS = frozenset({"after", "next_steps"})


def scan_lineages(
    project_dir: str | Path,
    category: str,
) -> list[tuple[list[str], Path]]:
    """Locate all computed lineages for *category* under *project_dir*.

    Scans the directory tree for folders named ``<category>_N`` that contain
    an ``_output`` subdirectory.  The lineage names are reconstructed from
    the relative path segments between *project_dir* and the matched folder.

    :param project_dir: MSNoise project root (contains ``project.yaml``).
    :param category:    Category name without set number, e.g. ``"stack"``.
    :returns: List of ``(lineage_names, step_dir)`` tuples, one per match.
              *lineage_names* is ordered from root to leaf (e.g.
              ``["global_1", "preprocess_1", "cc_1", "filter_1", "stack_1"]``).
              *step_dir* is the absolute :class:`~pathlib.Path` to the matched
              ``<category>_N`` folder.
    """
    root = Path(project_dir).resolve()
    results: list[tuple[list[str], Path]] = []

    for candidate in root.rglob(f"{category}_*"):
        if not candidate.is_dir():
            continue
        if not (candidate / "_output").is_dir():
            continue
        rel_parts = candidate.relative_to(root).parts
        lineage_names = list(rel_parts)
        results.append((lineage_names, candidate))

    results.sort(key=lambda t: t[0])
    return results


def build_params_from_project_yaml(
    project_yaml_path: str | Path,
    lineage_names: list[str],
):
    """Build an :class:`~msnoise.params.MSNoiseParams` from a ``project.yaml``.

    Reads *project_yaml_path* and extracts the config layers for each step in
    *lineage_names*.  The returned object has the same structure as one
    produced by :meth:`~msnoise.params.MSNoiseParams.from_yaml`.

    :param project_yaml_path: Path to ``project.yaml``.
    :param lineage_names:     Ordered list of step names, e.g.
                              ``["global_1", "preprocess_1", "stack_1"]``.
    :returns: :class:`~msnoise.params.MSNoiseParams` with one layer per
              category in *lineage_names*.
    """
    import yaml
    from obspy.core.util.attribdict import AttribDict
    from ..params import MSNoiseParams

    with open(project_yaml_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    p = MSNoiseParams()
    p._set_lineage_names(lineage_names)

    for step_name in lineage_names:
        step_doc = doc.get(step_name, {})
        config = {k: v for k, v in step_doc.items() if k not in _STEP_META_KEYS}
        category = step_name.rsplit("_", 1)[0]
        p._add_layer(category, AttribDict(config))

    return p
