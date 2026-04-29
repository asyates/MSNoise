"""Low-level I/O helpers for MSNoise project archives.

A *project archive* is a ``.tar.zst`` file containing a full MSNoise project
at a given pipeline level (preprocess / cc / stack / dvv …).  It is distinct
from an :class:`~msnoise.results.MSNoiseResult` *result bundle* (single
lineage, ``params.yaml`` + ``_output/``).

This module handles filesystem operations (extract, export, checksum, job
reconstruction).  Higher-level logic lives in :mod:`msnoise.project`.

Archive format
--------------

Every ``.tar.zst`` archive has this internal layout::

    meta.yaml              ← {entry_level, msnoise_version, created_at, project_name}
    MANIFEST.json          ← {relative_path: {sha256, size_bytes}} for every file
    project.yaml           ← full MSNoise config (importable)
    <lineage>/<step>/_output/...   ← pipeline outputs
    <lineage>/<step>/params.yaml   ← per-lineage params (enables DB-free access)

Paths inside the archive are relative to the project root, so extraction into
any directory produces a valid project tree.

``bundle_pointer.yaml`` format
-------------------------------

::

    msnoise_version_min: "2.0.1"
    created_at: "2026-04-01"
    levels:
      stack:
        description: "Stacked CCFs + refstacks for all filter sets"
        url: "https://ftp.seismology.be/msnoise/study/level_stack.tar.zst"
        sha256: "a1b2c3d4…"
        size_gb: 6.8
      dvv:
        description: "Final DVV aggregates and per-pair series"
        url: "https://zenodo.org/record/XXXXXXX/files/level_dvv.tar.zst"
        sha256: "d4e5f6…"
        size_gb: 0.3

``url`` must be a plain HTTPS URL; ``sha256`` is the hash of the
``.tar.zst`` file itself (printed by :func:`export_project` on completion).

Typical workflow
----------------

Export::

    from msnoise.core.project_io import export_project
    sha = export_project("/path/to/project", "stack", "/data/level_stack.tar.zst")
    # prints sha256 — paste into bundle_pointer.yaml

Import (Python)::

    from msnoise.core.project_io import import_project_archive
    root = import_project_archive("bundle_pointer.yaml", "stack", "./my_project")

Import (CLI)::

    msnoise project import --from bundle_pointer.yaml --level stack \\
        --project-dir ./my_project --with-jobs
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
    "wavelet":    ["**/wavelet_*/_output/**"],
    "dvv":        ["**/*_dvv/_output/**"],
}

#: Categories whose outputs are present in each entry level.
#: Used by :func:`reconstruct_jobs_from_filesystem` to know which steps
#: to scan when inserting ``flag=D`` jobs.
LEVEL_CATEGORIES: dict[str, list[str]] = {
    "preprocess": ["preprocess"],
    "cc":         ["cc"],
    "stack":      ["stack", "refstack"],
    "mwcs":       ["mwcs", "mwcs_dtt"],
    "stretching": ["stretching"],
    "wavelet":    ["wavelet", "wavelet_dtt"],
    "dvv":        ["mwcs_dtt_dvv", "stretching_dvv", "wavelet_dtt_dvv"],
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


# ---------------------------------------------------------------------------
# Job reconstruction helpers
# ---------------------------------------------------------------------------

def _pair_from_stem(stem: str) -> str:
    """Convert a NetCDF filename stem ``STA1_STA2`` to a job pair ``STA1:STA2``.

    SEED station identifiers use dots (e.g. ``BE.UCC..HHZ``) so the single
    underscore in the stem unambiguously separates the two stations.
    """
    return stem.replace("_", ":", 1)


def _days_from_nc(path: Path) -> list[str]:
    """Return unique date strings from the ``times`` coordinate of a NetCDF file.

    :returns: Sorted list of ``YYYY-MM-DD`` strings.
    """
    import xarray as xr
    ds = xr.open_dataset(str(path))
    try:
        times = ds["times"].values
    finally:
        ds.close()
    import numpy as np
    return sorted({str(t)[:10] for t in times})


def reconstruct_jobs_from_filesystem(session, schema, level: str, root: str | Path) -> int:
    """Insert ``flag=D`` jobs by scanning the extracted ``_output/`` tree.

    After ``msnoise db init --from-yaml`` the jobs table is empty.  This
    function synthetically populates it so that normal ``new_jobs``
    propagation generates the correct downstream ``flag=T`` jobs.

    :param session: SQLAlchemy session (DB must already be initialised).
    :param schema:  Return value of :func:`~msnoise.msnoise_table_def.declare_tables`.
    :param level:   Entry level string (key of :data:`LEVEL_CATEGORIES`).
    :param root:    Project root directory.
    :returns:       Total number of ``flag=D`` jobs inserted.
    :raises ValueError: if *level* is unknown.
    """
    import datetime
    from ..msnoise_table_def import WorkflowStep
    from ..core.workflow import _get_or_create_lineage_id

    if level not in LEVEL_CATEGORIES:
        raise ValueError(f"Unknown level {level!r}")

    root = Path(root).resolve()
    Job = schema.Job
    now = datetime.datetime.utcnow()
    total_inserted = 0

    for category in LEVEL_CATEGORIES[level]:
        matches = scan_lineages(root, category)
        for lineage_names, step_dir in matches:
            step_name = lineage_names[-1]

            # Resolve step_id
            step = (
                session.query(WorkflowStep)
                .filter(WorkflowStep.step_name == step_name)
                .first()
            )
            if step is None:
                import logging
                logging.getLogger("msnoise.project_io").warning(
                    f"WorkflowStep {step_name!r} not found in DB — skipping"
                )
                continue

            lineage_str = "/".join(lineage_names)
            lineage_id = _get_or_create_lineage_id(session, lineage_str)
            session.flush()

            output_dir = step_dir / "_output"
            job_tuples: list[tuple[str, str]] = []  # (pair, day)

            if category == "refstack":
                # REF files: _output/REF/<comp>/<STA1>_<STA2>.nc
                for nc in output_dir.rglob("*.nc"):
                    pair = _pair_from_stem(nc.stem)
                    job_tuples.append((pair, "REF"))

            elif category in ("mwcs_dtt_dvv", "stretching_dvv", "wavelet_dtt_dvv"):
                # DVV pair files: _output/<ms>/dvv_pairs_*.nc — pairs in `pair` dim
                import xarray as xr
                for nc in output_dir.rglob("dvv_pairs_*.nc"):
                    ds = xr.open_dataset(str(nc))
                    try:
                        pairs = [str(p) for p in ds["pair"].values]
                    finally:
                        ds.close()
                    for pair in pairs:
                        job_tuples.append((pair, "DVV"))

            elif category == "cc":
                # Daily CCFs: _output/daily/<comp>/<STA1>_<STA2>/<YYYY-MM-DD>.nc
                daily_root = output_dir / "daily"
                if daily_root.is_dir():
                    for nc in daily_root.rglob("*.nc"):
                        pair = _pair_from_stem(nc.parent.name)
                        day = nc.stem
                        job_tuples.append((pair, day))

            else:
                # stack, mwcs, wavelet, wavelet_dtt, mwcs_dtt, stretching:
                # _output/<ms>/<comp>/<STA1>_<STA2>.nc — days from `times` dim
                for nc in output_dir.rglob("*.nc"):
                    pair = _pair_from_stem(nc.stem)
                    try:
                        days = _days_from_nc(nc)
                    except Exception:
                        days = []
                    for day in days:
                        job_tuples.append((pair, day))

            # Deduplicate
            job_set = list(set(job_tuples))

            mappings = [
                {
                    "day":        day,
                    "pair":       pair,
                    "flag":       "D",
                    "jobtype":    step_name,
                    "step_id":    step.step_id,
                    "lineage_id": lineage_id,
                    "priority":   0,
                    "lastmod":    now,
                }
                for pair, day in job_set
            ]
            if mappings:
                session.bulk_insert_mappings(Job, mappings)
                total_inserted += len(mappings)

    session.commit()
    return total_inserted


def import_project_archive(
    pointer_path: str | Path,
    level: str,
    project_dir: str | Path,
) -> Path:
    """Download and extract a project archive from a ``bundle_pointer.yaml``.

    :param pointer_path: Path to ``bundle_pointer.yaml``.
    :param level:        Entry level to import (must be listed in the pointer).
    :param project_dir:  Destination directory (created if absent).
    :returns:            Absolute path to the extracted project root.
    :raises KeyError:    if *level* is not listed in ``bundle_pointer.yaml``.
    :raises ValueError:  if the downloaded archive fails the SHA-256 check.
    """
    import tempfile
    import urllib.request
    import yaml

    pointer_path = Path(pointer_path).resolve()
    with open(pointer_path, encoding="utf-8") as fh:
        pointer = yaml.safe_load(fh)

    levels = pointer.get("levels", {})
    if level not in levels:
        available = list(levels)
        raise KeyError(
            f"Level {level!r} not in bundle_pointer.yaml. "
            f"Available: {available}"
        )

    entry = levels[level]
    url = entry["url"]
    expected_sha = entry.get("sha256", "")

    # Download with progress
    import sys
    print(f"Downloading {url} …", flush=True)

    def _report(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r  {pct}%", end="", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    urllib.request.urlretrieve(url, str(tmp_path), reporthook=_report)
    print()  # newline after progress

    # Verify sha256
    actual_sha = file_sha256(tmp_path)
    if expected_sha and actual_sha != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch: expected {expected_sha!r}, got {actual_sha!r}"
        )

    root = extract_archive(tmp_path, project_dir)
    tmp_path.unlink(missing_ok=True)
    return root
