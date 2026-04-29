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


# ---------------------------------------------------------------------------
# Project archive export
# ---------------------------------------------------------------------------

#: Glob patterns (relative to project root) for each entry level.
#: Each pattern addresses the ``_output/`` directory of the relevant steps.
LEVEL_GLOBS: dict[str, list[str]] = {
    "preprocess": ["preprocess_*/_output/**"],
    "cc":         ["preprocess_*/cc_*/_output/**"],
    "stack":      ["**/filter_*/stack_*/_output/**",
                   "**/filter_*/refstack_*/_output/**"],
    "mwcs":       ["**/mwcs_*/_output/**", "**/mwcs_dtt_*/_output/**"],
    "stretching": ["**/stretching_*/_output/**"],
    "wavelet":    ["**/wct_*/_output/**", "**/wct_dtt_*/_output/**"],
    "dvv":        ["**/*_dvv/_output/**"],
}


def export_project(
    project_dir: str | Path,
    level: str,
    output_path: str | Path,
) -> str:
    """Export a project archive (``.tar.zst``) for the given entry *level*.

    Collects all ``_output/`` trees matching *level* (see :data:`LEVEL_GLOBS`),
    generates a ``params.yaml`` alongside each matched step directory, writes
    ``meta.yaml`` and ``MANIFEST.json``, and streams everything into a
    ``.tar.zst`` file.

    :param project_dir:  MSNoise project root (contains ``project.yaml``).
    :param level:        Entry level — one of the keys in :data:`LEVEL_GLOBS`.
    :param output_path:  Destination ``.tar.zst`` file path (created/overwritten).
    :returns:            Hex SHA-256 of the written archive (paste into
                         ``bundle_pointer.yaml``).
    :raises ValueError:  if *level* is not a recognised entry level.
    :raises FileNotFoundError: if ``project.yaml`` is absent from *project_dir*.
    """
    import datetime
    import io
    import json
    import tarfile
    import zstandard as zstd
    import yaml
    from obspy.core.util.attribdict import AttribDict

    if level not in LEVEL_GLOBS:
        raise ValueError(
            f"Unknown level {level!r}. Choose from: {list(LEVEL_GLOBS)}"
        )

    root = Path(project_dir).resolve()
    yaml_path = root / "project.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"project.yaml not found in {root}")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── collect matching _output dirs then gather files inside ───────────
    # The trailing /** glob only matches subdirs in Python ≥3.12, not files.
    # Strategy: find _output dirs via the level patterns (without trailing /**),
    # then rglob("*") inside each to collect actual files.
    output_dir_patterns = [p.rstrip("/**").rstrip("/") for p in LEVEL_GLOBS[level]]

    output_dirs: list[Path] = []
    for pattern in output_dir_patterns:
        for candidate in root.glob(pattern):
            if candidate.is_dir() and candidate.name == "_output":
                output_dirs.append(candidate)

    matched_files: list[Path] = []
    for odir in output_dirs:
        for f in odir.rglob("*"):
            if f.is_file():
                matched_files.append(f)
    matched_files = sorted(set(matched_files))

    # ── find unique step dirs and generate params.yaml per lineage ────────
    # Step dir = parent of _output dir.
    step_dirs: set[Path] = {odir.parent for odir in output_dirs}

    params_files: list[Path] = []
    for step_dir in sorted(step_dirs):
        params_path = step_dir / "params.yaml"
        lineage_names = list(step_dir.relative_to(root).parts)
        params = build_params_from_project_yaml(yaml_path, lineage_names)
        # output_folder will be re-anchored by project.list() at import time.
        layers = object.__getattribute__(params, "_layers")
        if "global" in layers:
            gd = dict(layers["global"])
            gd["output_folder"] = str(root)
            layers["global"] = AttribDict(gd)
        params.to_yaml(str(params_path))
        params_files.append(params_path)

    # ── files to archive: _output contents + params.yaml + project.yaml ──
    archive_files: list[Path] = (
        matched_files + params_files + [yaml_path]
    )
    archive_files = sorted(set(archive_files))

    # ── build MANIFEST and meta ───────────────────────────────────────────
    from .._version import version as msnoise_version  # type: ignore

    manifest: dict[str, dict] = {}
    for f in archive_files:
        rel = str(f.relative_to(root))
        sha = file_sha256(f)
        manifest[rel] = {"sha256": sha, "size_bytes": f.stat().st_size}

    meta = {
        "entry_level": level,
        "msnoise_version": msnoise_version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "project_name": root.name,
    }

    # ── stream to .tar.zst ────────────────────────────────────────────────
    archive_sha = hashlib.sha256()

    with open(output_path, "wb") as out_fh:
        cctx = zstd.ZstdCompressor(level=9)
        with cctx.stream_writer(out_fh, closefd=False) as compressor:
            with tarfile.open(fileobj=compressor, mode="w|") as tf:

                def _add_bytes(name: str, data: bytes) -> None:
                    buf = io.BytesIO(data)
                    ti = tarfile.TarInfo(name=name)
                    ti.size = len(data)
                    tf.addfile(ti, buf)

                _add_bytes("meta.yaml", yaml.dump(meta, default_flow_style=False).encode())
                _add_bytes("MANIFEST.json", json.dumps(manifest, indent=2).encode())

                for f in archive_files:
                    rel = str(f.relative_to(root))
                    tf.add(str(f), arcname=rel)

    # sha256 of the archive file itself
    archive_sha = file_sha256(output_path)
    return archive_sha
