"""MSNoiseProject — unified entry point for accessing MSNoise results.

:class:`MSNoiseProject` is the single entry point for reading MSNoise results
— regardless of whether data lives in a local live project, a project archive
on disk, or a paper from the MSNoise Reproducible Papers registry.

All three paths converge on the same API::

    # A — live project (cwd contains db.ini)
    from msnoise.project import MSNoiseProject
    project = MSNoiseProject.from_current()

    # B — local project archive
    project = MSNoiseProject.from_archive("level_stack.tar.zst")

    # C — MSNoise Reproducible Papers (auto-download)
    from msnoise.papers import MRP
    project = MRP().get_paper("2016_DePlaen_PitonDeLaFournaise").get_project("stack")

    # identical from here — all three paths
    for result in project.list("stack"):
        ds = result.get_ccf()


Project archives vs result bundles
------------------------------------

Two distinct archive types exist in MSNoise 2.x:

* **Project archive** (``.tar.zst``) — full multi-lineage project at a given
  *entry level*, containing all filter / stack branches.  Produced by
  ``msnoise project export``, consumed by ``msnoise project import`` or
  :meth:`MSNoiseProject.from_archive`.

* **Result bundle** (directory or ``.zip``) — single-lineage portable export:
  ``params.yaml`` + ``_output/``.  Produced and consumed by
  :meth:`~msnoise.results.MSNoiseResult.export_bundle` /
  :meth:`~msnoise.results.MSNoiseResult.from_bundle`.

Entry levels
------------

A project archive is created at a specific *entry level* — the lowest
pipeline step whose outputs are included.

================  ===========================================  ===========================
Level             What is bundled                              Resume from …
================  ===========================================  ===========================
``preprocess``    SDS waveform cache                           ``cc`` onwards
``cc``            Raw CCF NetCDFs                              ``stack`` + ``refstack``
``stack``         Stacked CCFs + reference stacks              ``mwcs``, ``stretching``, ``wavelet``
``mwcs``          MWCS + DTT outputs                           ``mwcs_dtt_dvv``
``stretching``    Stretching outputs                           ``stretching_dvv``
``wavelet``       WCT + WCT-DTT outputs                        ``wavelet_dtt_dvv``
``dvv``           Final dv/v aggregates + per-pair series      Notebooks only
================  ===========================================  ===========================

``stack`` and ``refstack`` outputs are always bundled together.

Exporting a project archive
----------------------------

Run from the project root after the pipeline has finished::

    msnoise project export --level stack --output /data/level_stack.tar.zst

The command prints the archive SHA-256 to paste into ``bundle_pointer.yaml``.
No database connection is needed — only the ``_output/`` tree and
``project.yaml`` are read.

Python equivalent::

    from msnoise.core.project_io import export_project
    sha = export_project("/path/to/project", "stack", "/data/level_stack.tar.zst")

Importing a project archive
----------------------------

Download, verify, extract, and initialise the database in one step::

    msnoise project import \\
        --from bundle_pointer.yaml \\
        --level stack \\
        --project-dir ./my_project \\
        --with-jobs

``--with-jobs`` reconstructs ``flag=D`` jobs from the ``_output/`` tree so
the pipeline can be resumed immediately afterwards::

    msnoise new_jobs --after stack

Reading results without a database
------------------------------------

:meth:`MSNoiseProject.list` is always filesystem-based — no database needed::

    project = MSNoiseProject.from_project_dir("/path/to/extracted")
    results = project.list("stack")

    for result in results:
        print(result.lineage_names)   # ['global_1', ..., 'stack_1']
        ds = result.get_ccf(component="ZZ", mov_stack=("1D", "1D"))

    # Traverse to child steps (folder scan, no DB)
    for result in results:
        for branch in result.branches():
            print(branch.category, branch.lineage_names[-1])

Resuming the pipeline
----------------------

To continue running the pipeline after importing an archive::

    # via CLI (recommended)
    msnoise project import --from bundle_pointer.yaml --level stack \\
        --project-dir ./my_project --with-jobs

    # via Python
    project = MSNoiseProject.from_archive("level_stack.tar.zst",
                                          project_dir="./my_project")
    project.init_db(with_jobs=True)
    db = project.db   # SQLAlchemy session now available
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
        _imported_levels: "list[str] | None" = None,
    ) -> None:
        self.project_dir: str = str(Path(project_dir).resolve())
        self._db = _db
        self._tmpdir = _tmpdir  # TemporaryDirectory kept alive if archive extracted to temp
        self._imported_levels = _imported_levels  # set when created via from_archive / import

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
        path: "str | Path | list",
        project_dir: "str | Path | None" = None,
    ) -> "MSNoiseProject":
        """Load a project from one or more ``.tar.zst`` project archives.

        If *project_dir* is ``None`` the archive(s) are extracted into a
        temporary directory kept alive for the lifetime of the returned object.
        Pass an explicit *project_dir* for a persistent extraction.

        When a list of archives is supplied they are all extracted into the
        **same** *project_dir* — their ``_output/`` trees never overlap, so
        the result is a composite project equivalent to having run every
        bundled level locally.

        :param path:        Path **or list of paths** to ``.tar.zst`` archive(s).
        :param project_dir: Destination directory.  ``None`` → auto temp dir.
        :returns:           :class:`MSNoiseProject` with ``_db=None``.
        """
        from .core.project_io import extract_archive

        _tmpdir = None
        if project_dir is None:
            _tmpdir = tempfile.TemporaryDirectory(prefix="msnoise_project_")
            project_dir = _tmpdir.name

        paths = [path] if isinstance(path, (str, Path)) else list(path)
        root = None
        for p in paths:
            root = extract_archive(p, project_dir)

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
        :raises NotImplementedError: if ``with_jobs=True`` but ``meta.yaml``
                                     is absent from the project directory.
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
            from .core.project_io import reconstruct_jobs_from_filesystem, LEVEL_CATEGORIES
            from .msnoise_table_def import declare_tables
            import yaml as _yaml
            _schema = declare_tables()

            # Determine which levels to reconstruct:
            # 1. Use _imported_levels if set (set by from_archive / import_project_archive)
            # 2. Fall back to meta.yaml (single-level import via CLI)
            if self._imported_levels:
                levels_to_reconstruct = self._imported_levels
            else:
                meta_path = os.path.join(self.project_dir, "meta.yaml")
                if not os.path.isfile(meta_path):
                    raise FileNotFoundError(
                        "meta.yaml not found — cannot determine entry level for "
                        "job reconstruction.  Ensure the project archive is intact."
                    )
                with open(meta_path, encoding="utf-8") as _fh:
                    meta = _yaml.safe_load(_fh)
                levels_to_reconstruct = [meta["entry_level"]]

            total = 0
            for lv in levels_to_reconstruct:
                n = reconstruct_jobs_from_filesystem(
                    self._db, _schema, level=lv, root=self.project_dir
                )
                total += n
            print(f"Reconstructed {total} flag=D jobs from filesystem "
                  f"(levels={levels_to_reconstruct}).")

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

            # Back-fill CSV defaults into every layer — params.yaml may
            # have been written with only_non_defaults=True, leaving keys
            # like maxlag absent.  Merge order: defaults < yaml values.
            import csv as _csv
            from pathlib import Path as _Path
            _cfg_dir = _Path(__file__).parent / "config"
            _type_map = {"float": float, "int": int, "bool": bool, "str": str}
            _layers = object.__getattribute__(params, "_layers")
            for _cat, _layer in list(_layers.items()):
                _csv_path = _cfg_dir / f"config_{_cat}.csv"
                if not _csv_path.exists():
                    continue
                with open(_csv_path, newline="", encoding="utf-8") as _fh:
                    _rows = list(_csv.DictReader(_fh))
                for _row in _rows:
                    _k = _row["name"]
                    if not hasattr(_layer, _k):
                        _v = _row["default"]
                        _cast = _type_map.get(_row.get("type", "str"), str)
                        try:
                            _layer[_k] = _cast(_v)
                        except (ValueError, TypeError):
                            _layer[_k] = _v

            # Construct MSNoiseResult directly — bypass from_bundle's
            # unconditional output_folder override.
            inst = MSNoiseResult.__new__(MSNoiseResult)
            inst._db = None
            inst._bundle_root = str(step_dir)
            inst._tmpdir = None
            # lineage_names comes from relative path parts — the archive
            # may have been rooted under the original output_folder basename
            # (e.g. "OUTPUT").  Strip any leading parts that aren't valid
            # step names (<word>_<digit>) so _step_prefix never sees them.
            import re as _re
            _step_pat = _re.compile(r".+_\d+$")
            clean_lineage = [n for n in lineage_names if _step_pat.match(n)]
            # Derive the actual output root by walking up len(clean_lineage)
            # levels from step_dir.  This handles archives that embed the
            # original output_folder name (e.g. level_dvv/OUTPUT/preprocess_1/…).
            actual_root = step_dir.parents[len(clean_lineage) - 1]
            inst.lineage_names = clean_lineage
            inst.params = params
            inst.output_folder = str(actual_root)
            inst.category = _step_prefix(clean_lineage[-1])
            inst._present_categories = frozenset(
                _step_prefix(n) for n in clean_lineage
            )
            results.append(inst)

        return results

    def get_stations(self) -> list:
        """Return station list as :class:`obspy.core.util.attribdict.AttribDict` objects.

        If a DB session is available (:meth:`from_current`), queries the
        ``Station`` table.  Otherwise parses ``project.yaml`` from
        :attr:`project_dir` — works in DB-free archive mode.

        Each item exposes ``net``, ``sta``, ``X``, ``Y``, ``altitude``,
        ``coordinates`` attributes, compatible with
        :func:`~msnoise.core.stations.get_interstation_distance`.
        """
        from obspy.core.util.attribdict import AttribDict

        if self._db is not None:
            from msnoise.api import get_stations as _get_stations
            return _get_stations(self._db)

        import yaml
        yaml_path = Path(self.project_dir) / "project.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"project.yaml not found in {self.project_dir}. "
                "Connect a DB via init_db() or ensure project.yaml exists."
            )
        with open(yaml_path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        return [AttribDict(s) for s in doc.get("stations", [])]

    def get_distance(self, pair: str) -> float:
        """Return interstation distance in km for *pair*.

        :param pair: Station pair in ``"NET.STA.LOC:NET.STA.LOC"`` format.
        :returns: Distance in kilometres.
        :raises KeyError: if either station is not found.
        """
        from msnoise.core.stations import get_interstation_distance

        sta1_id, sta2_id = pair.split(":")
        n1, s1 = sta1_id.split(".")[:2]
        n2, s2 = sta2_id.split(".")[:2]

        stations = {f"{st.net}.{st.sta}": st for st in self.get_stations()}
        key1, key2 = f"{n1}.{s1}", f"{n2}.{s2}"
        if key1 not in stations:
            raise KeyError(f"Station {key1!r} not found in project")
        if key2 not in stations:
            raise KeyError(f"Station {key2!r} not found in project")

        st1, st2 = stations[key1], stations[key2]
        coords = getattr(st1, "coordinates", "DEG")
        return get_interstation_distance(st1, st2, coordinates=coords)

