"""MSNoise Reproducible Papers (MRP) client.

Provides programmatic access to the `MSNoise Reproducible Papers
<https://github.com/ROBelgium/MSNoise_Reproducible_Papers>`_ registry — a
curated collection of ``project.yaml`` files and optional data bundles that
reproduce published studies using MSNoise.

Quick start::

    from msnoise.papers import MRP

    mrp = MRP()
    mrp.list_papers()

    paper = mrp.get_paper("2016_DePlaen_PitonDeLaFournaise")
    paper.info()

    # Downloads the archive on first call; cached locally afterwards.
    project = paper.get_project("stack")
    for result in project.list("stack"):
        ds = result.get_ccf()

The returned :class:`~msnoise.project.MSNoiseProject` is identical to one
obtained via :meth:`~msnoise.project.MSNoiseProject.from_archive` — all
``get_*`` methods work without a database connection.

Browsing available papers
--------------------------

:meth:`MRP.list_papers` prints a table of all papers in the registry::

    mrp = MRP()
    mrp.list_papers()
    # ID                                    Year  Net       Levels          ✓
    # 2016_DePlaen_PitonDeLaFournaise       2016  PF......  stack, dvv      ✅

Loading a paper
---------------

::

    paper = mrp.get_paper("2016_DePlaen_PitonDeLaFournaise")
    paper.info()
    # Paper:   2016_DePlaen_PitonDeLaFournaise
    # journal_abbrev: GRL
    # ...
    # bundle_levels_available: ['stack', 'dvv']

Papers with multiple datasets (e.g. two volcanoes) expose multiple project
files.  Pass ``project=`` to disambiguate::

    paper = mrp.get_paper("2023_Yates_PitonRuapehu")
    project_pdf     = paper.get_project("dvv", project="pdf")
    project_ruapehu = paper.get_project("dvv", project="ruapehu")

Cache management
-----------------

Downloaded archives are stored in the platform user-cache directory
(``~/.cache/msnoise-mrp/`` on Linux).  To free space::

    mrp.clear_cache("2016_DePlaen_PitonDeLaFournaise")  # one paper
    mrp.clear_cache()                                    # all archives

Registry metadata and small paper files are never deleted by
:meth:`MRP.clear_cache`.  To force a fresh registry download::

    mrp = MRP(force_refresh=True)

Contributing a paper
---------------------

See the `CONTRIBUTING guide
<https://github.com/ROBelgium/MSNoise_Reproducible_Papers/blob/main/CONTRIBUTING.md>`_
in the registry repository.  In brief:

1. Fork the repo, create ``papers/<YYYY_Author_Title>/``
2. Add ``project.yaml``, ``citation.bib``, ``meta.yaml``, ``README.md``
3. Run ``python scripts/update_registry.py && python scripts/update_readme.py``
4. Open a PR — CI validates schemas and runs ``msnoise db init`` on every
   ``project*.yaml``
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LevelNotAvailable(KeyError):
    """Raised when a requested bundle level is absent from ``bundle_pointer.yaml``."""


class AmbiguousProject(ValueError):
    """Raised when a paper has multiple project files and no ``project=`` kwarg
    was supplied to :meth:`MRPPaper.get_project`."""


# ---------------------------------------------------------------------------
# Registry URL
# ---------------------------------------------------------------------------

_REGISTRY_BASE = (
    "https://raw.githubusercontent.com/ROBelgium/"
    "MSNoise_Reproducible_Papers/main"
)
_REGISTRY_URL = f"{_REGISTRY_BASE}/registry.yaml"


# ---------------------------------------------------------------------------
# MRP
# ---------------------------------------------------------------------------

class MRP:
    """Client for the MSNoise Reproducible Papers registry.

    :param cache_dir:      Local directory used to cache downloaded files.
                           Defaults to the platform user-cache directory for
                           ``"msnoise-mrp"`` (e.g. ``~/.cache/msnoise-mrp``
                           on Linux).
    :param force_refresh:  If ``True``, re-download ``registry.yaml`` even if
                           a cached copy exists.  Downloaded paper archives are
                           *never* re-downloaded; use :meth:`clear_cache` to
                           force a fresh download.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        force_refresh: bool = False,
    ) -> None:
        import platformdirs

        if cache_dir is None:
            cache_dir = platformdirs.user_cache_dir("msnoise-mrp")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._registry: dict = self._fetch_registry(force_refresh)

    # ------------------------------------------------------------------ #
    # Registry                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_registry(self, force_refresh: bool) -> dict:
        """Download (or load from cache) ``registry.yaml``."""
        import yaml

        registry_cache = self.cache_dir / "registry.yaml"
        if force_refresh or not registry_cache.exists():
            self._download(_REGISTRY_URL, registry_cache)
        with open(registry_cache, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        return doc

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        """Download *url* to *dest* using urllib (no extra deps).

        Supports ``http://``, ``https://``, and ``ftp://`` URLs (Python's
        urllib includes a native FTP handler).  A progress line is printed
        to stdout for files larger than 1 MB.
        """
        import urllib.request
        import sys

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")

        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / 1e6
                total_mb = total_size / 1e6
                sys.stdout.write(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB")
            else:
                sys.stdout.write(f"\r  {downloaded / 1e6:.1f} MB downloaded")
            sys.stdout.flush()

        try:
            urllib.request.urlretrieve(url, str(tmp), reporthook=_reporthook)
            sys.stdout.write("\n")
            tmp.replace(dest)
        except Exception:
            sys.stdout.write("\n")
            tmp.unlink(missing_ok=True)
            raise

    def _papers(self) -> list[dict]:
        return self._registry.get("papers", [])

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def list_papers(self) -> None:
        """Pretty-print a table of available papers."""
        papers = self._papers()
        if not papers:
            print("No papers found in registry.")
            return

        header = f"{'ID':<50} {'Year':>4}  {'Net':<8}  {'Levels':<30}  {'✓'}"
        print(header)
        print("-" * len(header))
        for p in papers:
            levels = ", ".join(p.get("levels_available", []))
            validated = "✅" if p.get("validated") else "  "
            print(
                f"{p['id']:<50} {p.get('year', ''):>4}  "
                f"{p.get('network', ''):.<8}  {levels:<30}  {validated}"
            )

    def get_paper(self, paper_id: str) -> "MRPPaper":
        """Fetch a paper's metadata and return an :class:`MRPPaper` object.

        Downloads ``project*.yaml``, ``meta.yaml``, and ``bundle_pointer.yaml``
        (if present) from the registry into the local cache.

        :param paper_id: Folder name in the registry, e.g.
                         ``"2016_DePlaen_PitonDeLaFournaise"``.
        :raises KeyError: if *paper_id* is not listed in the registry.
        """
        ids = [p["id"] for p in self._papers()]
        if paper_id not in ids:
            raise KeyError(
                f"Paper {paper_id!r} not in registry.  "
                f"Available: {ids}"
            )

        paper_cache = self.cache_dir / paper_id
        paper_cache.mkdir(parents=True, exist_ok=True)

        base = f"{_REGISTRY_BASE}/papers/{paper_id}"

        # Always refresh small metadata files
        for fname in ("meta.yaml", "bundle_pointer.yaml"):
            dest = paper_cache / fname
            try:
                self._download(f"{base}/{fname}", dest)
            except Exception:
                pass  # bundle_pointer.yaml may not exist for all papers

        # Discover and cache project*.yaml files listed in registry
        meta_entry = next(p for p in self._papers() if p["id"] == paper_id)
        project_files: list[str] = meta_entry.get("project_files", ["project.yaml"])
        for pf in project_files:
            dest = paper_cache / pf
            if not dest.exists():
                try:
                    self._download(f"{base}/{pf}", dest)
                except Exception:
                    pass
        # Fallback: always try project.yaml
        project_yaml = paper_cache / "project.yaml"
        if not project_yaml.exists():
            try:
                self._download(f"{base}/project.yaml", project_yaml)
            except Exception:
                pass

        return MRPPaper(paper_id, paper_cache, self)

    def clear_cache(self, paper_id: Optional[str] = None) -> None:
        """Delete downloaded archives from the local cache.

        Registry cache (``registry.yaml``) and metadata files are *never*
        deleted — only ``.tar.zst`` archives.

        :param paper_id: Delete archives for this paper only.  ``None``
                         (default) deletes all paper archives.
        """
        total_freed = 0
        if paper_id is not None:
            dirs = [self.cache_dir / paper_id]
        else:
            dirs = [
                self.cache_dir / p["id"]
                for p in self._papers()
                if (self.cache_dir / p["id"]).is_dir()
            ]

        for paper_dir in dirs:
            for f in paper_dir.rglob("*.tar.zst"):
                size = f.stat().st_size
                f.unlink()
                total_freed += size
                print(f"  Deleted {f.relative_to(self.cache_dir)}")

        mb = total_freed / (1024 ** 2)
        print(f"Freed {mb:.1f} MB")


# ---------------------------------------------------------------------------
# MRPPaper
# ---------------------------------------------------------------------------

class MRPPaper:
    """Represents a single paper in the MRP registry.

    Not constructed directly — obtain via :meth:`MRP.get_paper`.
    """

    def __init__(self, paper_id: str, cache_dir: Path, mrp: MRP) -> None:
        self.paper_id = paper_id
        self._cache_dir = cache_dir
        self._mrp = mrp
        self._meta: dict = self._load_yaml("meta.yaml") or {}
        self._pointer: dict = self._load_yaml("bundle_pointer.yaml") or {}

    def _load_yaml(self, fname: str) -> dict | None:
        import yaml
        path = self._cache_dir / fname
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def project_yaml(self) -> dict:
        """The raw parsed ``project.yaml`` for the default project."""
        import yaml
        path = self._cache_dir / "project.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"project.yaml not found in cache for {self.paper_id!r}. "
                "Try mrp.get_paper() again to refresh."
            )
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    @property
    def projects(self) -> dict[str, str]:
        """Map of project name → absolute path to ``project*.yaml`` in cache.

        A paper with a single ``project.yaml`` has key ``"default"``.
        Papers with multiple datasets have keys like ``"pdf"``, ``"ruapehu"``,
        derived from ``project_<name>.yaml`` filenames.
        """
        result: dict[str, str] = {}
        for f in sorted(self._cache_dir.glob("project*.yaml")):
            stem = f.stem  # e.g. "project" or "project_pdf"
            if stem == "project":
                key = "default"
            else:
                key = stem[len("project_"):]  # strip "project_" prefix
            result[key] = str(f)
        return result

    # ------------------------------------------------------------------ #
    # Methods                                                              #
    # ------------------------------------------------------------------ #

    def info(self) -> None:
        """Print metadata for this paper."""
        print(f"Paper:   {self.paper_id}")
        for k, v in self._meta.items():
            print(f"  {k}: {v}")
        levels = list(self._pointer.get("levels", {}).keys())
        print(f"  bundle_levels_available: {levels or '(none)'}")

    def get_project(
        self,
        level: "str | list[str]",
        project: str = "default",
    ) -> "msnoise.project.MSNoiseProject":  # type: ignore[name-defined]
        """Download archive(s) for *level* and return an :class:`~msnoise.project.MSNoiseProject`.

        Archives are downloaded once and cached permanently; subsequent calls
        return immediately (or skip already-extracted levels).  Use
        :meth:`MRP.clear_cache` to force a fresh download.

        :param level:   Entry level(s) to download.  Pass a single string
                        (e.g. ``"stack"``), a list (``["stack", "dvv"]``), or
                        ``"all"`` to download every level in ``bundle_pointer.yaml``.
                        All archives are extracted into the **same** directory.
        :param project: Project name for papers with multiple datasets.
                        Omit (or ``"default"``) for single-project papers.
        :raises LevelNotAvailable: if a requested level is absent.
        :raises AmbiguousProject:  if multiple projects exist and *project* was
                                   not specified.
        :raises FileNotFoundError: if no ``bundle_pointer.yaml`` exists.
        """
        from .project import MSNoiseProject

        # Resolve project yaml
        projs = self.projects
        if not projs:
            raise FileNotFoundError(
                f"No project*.yaml found in cache for {self.paper_id!r}."
            )
        if project == "default" and len(projs) > 1 and "default" not in projs:
            raise AmbiguousProject(
                f"Paper {self.paper_id!r} has multiple projects: "
                f"{list(projs)}.  Pass project=<name> explicitly."
            )
        if project not in projs:
            raise KeyError(
                f"Project {project!r} not found.  Available: {list(projs)}"
            )

        # Check level availability
        available_levels = self._pointer.get("levels", {})
        if not available_levels:
            raise FileNotFoundError(
                f"No bundle_pointer.yaml found for {self.paper_id!r}."
            )

        # Resolve level list
        if level == "all":
            levels_to_get = list(available_levels)
        elif isinstance(level, str):
            levels_to_get = [level]
        else:
            levels_to_get = list(level)

        missing = [lv for lv in levels_to_get if lv not in available_levels]
        if missing:
            raise LevelNotAvailable(
                f"Level(s) {missing} not available for {self.paper_id!r}. "
                f"Available: {list(available_levels)}"
            )

        # Build a stable extract dir name from the level set.
        # Only use "all" when the caller literally passed level="all" —
        # not when a single-level paper happens to match all available levels.
        if level == "all":
            level_tag = "all"
        elif len(levels_to_get) == 1:
            level_tag = levels_to_get[0]
        else:
            level_tag = "+".join(sorted(levels_to_get))

        extract_dir = (
            self._mrp.cache_dir
            / self.paper_id
            / project
            / f"level_{level_tag}"
        )

        # Skip download entirely if already extracted
        if (extract_dir / "project.yaml").exists():
            proj = MSNoiseProject.from_project_dir(extract_dir)
            proj._imported_levels = levels_to_get
            return proj

        # Download + extract each level into the same extract_dir
        from .core.project_io import extract_archive, file_sha256

        for lv in levels_to_get:
            entry = available_levels[lv]
            url = entry["url"]
            expected_sha = entry.get("sha256", "")

            archive_name = f"level_{lv}_{expected_sha[:12] if expected_sha else 'nosha'}.tar.zst"
            archive_path = self._cache_dir / archive_name

            if not archive_path.exists():
                print(f"Downloading level={lv!r}: {url} …")
                self._mrp._download(url, archive_path)

                if expected_sha:
                    actual = file_sha256(archive_path)
                    if actual != expected_sha:
                        archive_path.unlink(missing_ok=True)
                        raise ValueError(
                            f"SHA-256 mismatch for {self.paper_id!r} level={lv!r}: "
                            f"expected {expected_sha!r}, got {actual!r}"
                        )

            extract_archive(archive_path, extract_dir)

        proj = MSNoiseProject.from_project_dir(extract_dir)
        proj._imported_levels = levels_to_get
        return proj
