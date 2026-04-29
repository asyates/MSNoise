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
        if self._db is None:
            raise RuntimeError(
                "No database connection.  Call init_db() first, or load the "
                "project with from_current() which connects automatically."
            )
        return self._db

    # ------------------------------------------------------------------ #
    # DB initialisation                                                   #
    # ------------------------------------------------------------------ #

    def init_db(self, with_jobs: bool = False) -> None:
        """Initialise the project database from ``project.yaml``.

        Runs ``msnoise db init --from-yaml project.yaml`` in
        :attr:`project_dir`, then connects to the created database.

        :param with_jobs: If ``True``, also reconstruct ``flag=D`` jobs by
                          scanning the extracted ``_output/`` tree.  Only
                          needed when continuing the pipeline after importing
                          a project archive.
        :raises NotImplementedError: ``with_jobs=True`` is reserved for P4.
        """
        import subprocess
        from .core.db import connect

        yaml_path = os.path.join(self.project_dir, "project.yaml")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(
                f"project.yaml not found in {self.project_dir}"
            )

        subprocess.run(
            ["msnoise", "db", "init", "--from-yaml", yaml_path],
            cwd=self.project_dir,
            check=True,
        )

        inifile = os.path.join(self.project_dir, "db.ini")
        self._db = connect(inifile=inifile)

        if with_jobs:
            from .core.project_io import reconstruct_jobs_from_filesystem
            from .msnoise_table_def import declare_tables
            import yaml as _yaml
            meta_path = os.path.join(self.project_dir, "meta.yaml")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    "meta.yaml not found — cannot determine entry level for "
                    "job reconstruction.  Pass level= explicitly or ensure the "
                    "project archive is intact."
                )
            with open(meta_path, encoding="utf-8") as _fh:
                meta = _yaml.safe_load(_fh)
            level = meta["entry_level"]
            _schema = declare_tables()
            n = reconstruct_jobs_from_filesystem(self._db, _schema, level=level, root=self.project_dir)
            print(f"Reconstructed {n} flag=D jobs from filesystem (level={level!r}).")

    # ------------------------------------------------------------------ #
    # Result access                                                       #
    # ------------------------------------------------------------------ #

    def list(self, category: str) -> list:
        """Return all computed :class:`~msnoise.results.MSNoiseResult` objects
        for *category*.

        Always filesystem-based — no database required.  Scans
        :attr:`project_dir` for ``<category>_N`` directories that contain an
        ``_output/`` subdirectory.  A ``params.yaml`` is written alongside
        ``_output/`` on first access (Option A) and reused on subsequent calls.

        The returned objects share the same interface as those obtained via
        :meth:`~msnoise.results.MSNoiseResult.from_bundle` — all ``get_*``
        methods work immediately.  :meth:`~msnoise.results.MSNoiseResult.branches`
        uses a folder scan and is fully functional.

        :param category: Category name without set number, e.g. ``"stack"``.
        :returns: List of :class:`~msnoise.results.MSNoiseResult` (``_db=None``),
                  sorted by lineage path.
        :raises FileNotFoundError: if ``project.yaml`` is absent from
                                   :attr:`project_dir`.
        """
        from .core.project_io import scan_lineages, build_params_from_project_yaml
        from .results import MSNoiseResult, _step_prefix
        from obspy.core.util.attribdict import AttribDict

        project_dir = Path(self.project_dir)
        yaml_path = project_dir / "project.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"project.yaml not found in {self.project_dir}. "
                "Use from_current() for a live project without project.yaml."
            )

        matches = scan_lineages(project_dir, category)
        results = []

        for lineage_names, step_dir in matches:
            params_path = step_dir / "params.yaml"

            if not params_path.exists():
                params = build_params_from_project_yaml(yaml_path, lineage_names)
                # Embed project root as output_folder so _branches_from_folders
                # resolves child paths correctly via lineage_names.
                layers = object.__getattribute__(params, "_layers")
                if "global" in layers:
                    global_dict = dict(layers["global"])
                    global_dict["output_folder"] = str(project_dir)
                    layers["global"] = AttribDict(global_dict)
                params.to_yaml(str(params_path))
            else:
                from .params import MSNoiseParams
                params = MSNoiseParams.from_yaml(str(params_path))
                # Re-anchor output_folder to the live project_dir (path may
                # have moved since params.yaml was written).
                layers = object.__getattribute__(params, "_layers")
                if "global" in layers:
                    global_dict = dict(layers["global"])
                    global_dict["output_folder"] = str(project_dir)
                    layers["global"] = AttribDict(global_dict)

            # Construct MSNoiseResult directly — bypass from_bundle's
            # unconditional output_folder override.
            inst = MSNoiseResult.__new__(MSNoiseResult)
            inst._db = None
            inst._bundle_root = str(step_dir)
            inst._tmpdir = None
            inst.lineage_names = list(lineage_names)
            inst.params = params
            inst.output_folder = str(project_dir)
            inst.category = _step_prefix(lineage_names[-1])
            inst._present_categories = frozenset(
                _step_prefix(n) for n in lineage_names
            )
            results.append(inst)

        return results
