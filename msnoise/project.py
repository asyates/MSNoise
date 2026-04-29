"""MSNoiseProject — unified entry point for accessing MSNoise results.

Provides a single object through which :class:`~msnoise.results.MSNoiseResult`
objects are obtained, regardless of whether the data comes from a local live
project, an extracted project archive, or a paper downloaded from the MSNoise
Reproducible Papers registry.

Typical usage::

    # A — live project (cwd contains msnoise.ini / db.ini)
    project = MSNoiseProject.from_current()

    # B — project archive on disk
    project = MSNoiseProject.from_archive("level_stack.tar.zst")

    # C — MSNoise Reproducible Papers registry (see msnoise.papers)
    from msnoise.papers import MRP
    project = MRP().get_paper("2016_DePlaen_PitonDeLaFournaise").get_project("stack")

    # identical from here — all three paths
    for result in project.list("stack"):
        ccfs = result.get_ccf()
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


class MSNoiseProject:
    """Unified entry point for accessing MSNoise results.

    Attributes
    ----------
    project_dir : str
        Absolute path to the project root directory.  Contains
        ``project.yaml`` and (after :meth:`init_db`) ``db.ini``.
    """

    def __init__(
        self,
        project_dir: str | Path,
        _db=None,
        _tmpdir=None,
    ) -> None:
        self.project_dir: str = str(Path(project_dir).resolve())
        self._db = _db
        self._tmpdir = _tmpdir  # TemporaryDirectory kept alive if archive extracted to temp

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_current(cls, project_dir: str | Path = ".") -> "MSNoiseProject":
        """Load a live project from *project_dir* (default: current directory).

        Reads ``db.ini`` and connects to the existing database.  The returned
        object has ``_db`` populated and is ready for pipeline operations as
        well as result access.

        :param project_dir: Path containing ``db.ini``.  Defaults to ``"."``.
        :raises FileNotFoundError: if ``db.ini`` is absent from *project_dir*.
        """
        from .core.db import connect

        project_dir = Path(project_dir).resolve()
        inifile = project_dir / "db.ini"
        if not inifile.exists():
            raise FileNotFoundError(
                f"db.ini not found in {project_dir}. "
                "Is this an MSNoise project directory?"
            )
        db = connect(inifile=str(inifile))
        return cls(project_dir, _db=db)

    @classmethod
    def from_archive(
        cls,
        path: str | Path,
        project_dir: str | Path | None = None,
    ) -> "MSNoiseProject":
        """Load a project from a ``.tar.zst`` project archive.

        If *project_dir* is ``None`` the archive is extracted into a temporary
        directory that is kept alive for the lifetime of the returned object.
        Pass an explicit *project_dir* for a persistent extraction.

        :param path:        Path to the ``.tar.zst`` archive.
        :param project_dir: Destination directory for extraction.  Created if
                            absent.  ``None`` → auto temporary directory.
        :returns:           :class:`MSNoiseProject` with ``_db=None``.
        """
        from .core.project_io import extract_archive

        _tmpdir = None
        if project_dir is None:
            _tmpdir = tempfile.TemporaryDirectory(prefix="msnoise_project_")
            project_dir = _tmpdir.name

        root = extract_archive(path, project_dir)
        return cls(root, _tmpdir=_tmpdir)

    @classmethod
    def from_project_dir(cls, project_dir: str | Path) -> "MSNoiseProject":
        """Point at an already-extracted project directory (no DB).

        Use when the archive has already been extracted manually or by a
        previous call to :meth:`from_archive` with a persistent *project_dir*.

        :param project_dir: Path containing ``project.yaml``.
        :raises FileNotFoundError: if ``project.yaml`` is absent.
        """
        project_dir = Path(project_dir).resolve()
        if not (project_dir / "project.yaml").exists():
            raise FileNotFoundError(
                f"project.yaml not found in {project_dir}. "
                "Is this an extracted MSNoise project archive?"
            )
        return cls(project_dir)

    # ------------------------------------------------------------------ #
    # DB access                                                            #
    # ------------------------------------------------------------------ #

    @property
    def db(self):
        """SQLAlchemy session.  Raises :exc:`RuntimeError` if not initialised.

        Call :meth:`init_db` first, or use :meth:`from_current` which
        connects automatically.
        """
