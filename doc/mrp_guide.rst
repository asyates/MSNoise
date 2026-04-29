.. _mrp_guide:

*****************************************************
Project archives & the Reproducible Papers registry
*****************************************************

This page explains how to **share**, **archive**, and **re-use** MSNoise
projects — and how to access peer-reviewed studies through the
`MSNoise Reproducible Papers <https://github.com/ROBelgium/MSNoise_Reproducible_Papers>`_
(MRP) registry.

.. contents:: Contents
   :local:
   :depth: 2


Concepts
========

Two distinct archive types exist in MSNoise 2.x.  Make sure you use the
right one:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Name
     - What it is
     - When to use it
   * - **Project archive**
       (``.tar.zst``)
     - Full multi-lineage project at a given *entry level* — may contain
       dozens of pair files across many filter / stack branches.
     - Sharing a whole study (HPC → laptop, paper reproducibility).
   * - **Result bundle**
       (directory or ``.zip``)
     - Single-lineage portable export: ``params.yaml`` + ``_output/``.
       Produced by :meth:`~msnoise.results.MSNoiseResult.export_bundle`.
     - Sharing one specific lineage result with a collaborator.

CLI mapping:

.. code-block:: sh

    # project archive
    msnoise project export --level stack --output level_stack.tar.zst
    msnoise project import --from bundle_pointer.yaml --level stack

    # result bundle (unchanged)
    # MSNoiseResult.export_bundle() / from_bundle()


Entry levels
============

A project archive is created at a specific *entry level* — the lowest
pipeline step whose outputs are included.  Users who receive the archive can
resume the pipeline from that point.

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Level
     - What is bundled
     - Lets the user run from …
   * - ``preprocess``
     - SDS waveform cache
     - ``cc`` onwards
   * - ``cc``
     - Raw CCF NetCDFs
     - ``stack`` + ``refstack``
   * - ``stack``
     - Stacked CCFs + reference stacks
     - ``mwcs``, ``stretching``, ``wavelet``
   * - ``mwcs``
     - MWCS + DTT outputs
     - ``mwcs_dtt_dvv``
   * - ``stretching``
     - Stretching outputs
     - ``stretching_dvv``
   * - ``wavelet``
     - WCT + WCT-DTT outputs
     - ``wavelet_dtt_dvv``
   * - ``dvv``
     - Final dv/v aggregates + per-pair series
     - Notebooks only (pipeline complete)

``stack`` and ``refstack`` outputs are always bundled together — both are
required for all downstream steps.

A paper typically publishes **two levels**: e.g. ``stack`` (cheapest
recompute entry) and ``dvv`` (figures-only).


Exporting a project archive
===========================

Run ``msnoise project export`` from the project root.  No database connection
is needed after the pipeline has finished — only the ``_output/`` tree and
``project.yaml`` are read.

.. code-block:: sh

    cd /path/to/my_project
    msnoise project export --level stack --output /data/level_stack.tar.zst

Options:

.. code-block:: text

    --level     preprocess | cc | stack | mwcs | stretching | wavelet | dvv
    -o/--output Destination .tar.zst file
    --project-dir  Project root (default: current directory)

The command prints the archive SHA-256 on completion::

    Done.  Archive SHA-256: a1b2c3d4…
    Paste into bundle_pointer.yaml:
      sha256: "a1b2c3d4…"

**What goes into the archive**

The archive contains, relative to the project root:

- ``project.yaml`` — full configuration (importable)
- ``meta.yaml`` — entry level, MSNoise version, timestamp
- ``MANIFEST.json`` — SHA-256 + size for every file (integrity check)
- One ``params.yaml`` alongside every included step directory (enables
  :class:`~msnoise.project.MSNoiseProject` to work without a database)
- All ``_output/`` trees for the requested level

Archive format is ``.tar.zst`` (zstandard level 9) for good compression
and fast streaming extraction.

**Python API**

.. code-block:: python

    from msnoise.core.project_io import export_project

    sha = export_project("/path/to/project", level="stack",
                         output_path="/data/level_stack.tar.zst")
    print(sha)   # paste into bundle_pointer.yaml


Importing a project archive
===========================

``msnoise project import`` downloads an archive from a URL listed in a
``bundle_pointer.yaml``, verifies its SHA-256, extracts it, and initialises
the database.

.. code-block:: sh

    msnoise project import \
        --from bundle_pointer.yaml \
        --level stack \
        --project-dir ./my_project \
        --with-jobs

Options:

.. code-block:: text

    --from       Path to bundle_pointer.yaml
    --level      Entry level to download
    --project-dir  Destination directory (created if absent; default: .)
    --with-jobs  Reconstruct flag=D jobs from the extracted _output/ tree
                 so the pipeline can be resumed immediately

After ``--with-jobs``, run the standard propagation command to generate
downstream jobs::

    msnoise new_jobs --after stack

**``bundle_pointer.yaml`` format**

.. code-block:: yaml

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
``.tar.zst`` file itself.

**Python API**

.. code-block:: python

    from msnoise.core.project_io import import_project_archive

    root = import_project_archive(
        pointer_path="bundle_pointer.yaml",
        level="stack",
        project_dir="./my_project",
    )
    # root is the extracted project directory


Reading results without a database
===================================

Once a project archive has been extracted (or imported), use
:class:`~msnoise.project.MSNoiseProject` to access results without any
database connection.  This is the recommended pattern on laptops and in
Jupyter notebooks.

.. code-block:: python

    from msnoise.project import MSNoiseProject

    # From an extracted directory
    project = MSNoiseProject.from_project_dir("/path/to/extracted")

    # Or directly from the .tar.zst (extracted to a temp dir automatically)
    project = MSNoiseProject.from_archive("level_stack.tar.zst")

    # List all computed stack lineages
    results = project.list("stack")
    print(len(results), "stack lineage(s) found")

    for result in results:
        print(result.lineage_names)  # e.g. ['global_1', ..., 'stack_1']
        ds = result.get_ccf(component="ZZ", mov_stack=("1D", "1D"))

    # Traverse to child steps (no DB needed — folder scan)
    for result in results:
        for branch in result.branches():
            print(branch.category, branch.lineage_names[-1])

Three entry paths are available, all returning an identical object:

.. code-block:: python

    # A — live project (cwd contains db.ini)
    project = MSNoiseProject.from_current()

    # B — local project archive
    project = MSNoiseProject.from_archive("level_stack.tar.zst")

    # C — MSNoise Reproducible Papers (auto-download)
    from msnoise.papers import MRP
    project = MRP().get_paper("2016_DePlaen_PitonDeLaFournaise").get_project("stack")

Resuming the pipeline
=====================

To continue running the pipeline after importing an archive, you need a
database::

    # Either via the CLI (recommended)
    msnoise project import --from bundle_pointer.yaml --level stack \
        --project-dir ./my_project --with-jobs

    # Or via Python
    project = MSNoiseProject.from_archive("level_stack.tar.zst",
                                          project_dir="./my_project")
    project.init_db(with_jobs=True)
    db = project.db   # SQLAlchemy session, now available

    # Then propagate jobs and run
    # msnoise new_jobs --after stack
    # msnoise run stack


The MSNoise Reproducible Papers registry
=========================================

The `MSNoise Reproducible Papers <https://github.com/ROBelgium/MSNoise_Reproducible_Papers>`_
(MRP) repository is a curated collection of ``project.yaml`` files (and
optional data bundles) for published studies.  The :mod:`msnoise.papers`
module provides a Python client.

Browsing available papers
--------------------------

.. code-block:: python

    from msnoise.papers import MRP

    mrp = MRP()
    mrp.list_papers()

Output::

    ID                                                  Year  Net       Levels                          ✓
    ---------------------------------------------------------------------------------------------------------------
    2016_DePlaen_PitonDeLaFournaise                     2016  PF......  stack, dvv                      ✅
    2019_DePlaen_Etna                                   2019  IV......  stack                              ...

Loading a paper
---------------

.. code-block:: python

    paper = mrp.get_paper("2016_DePlaen_PitonDeLaFournaise")
    paper.info()
    # Paper:   2016_DePlaen_PitonDeLaFournaise
    # journal_abbrev: GRL
    # network: PF
    # ...
    # bundle_levels_available: ['stack', 'dvv']

Accessing results
-----------------

.. code-block:: python

    # Downloads the archive on first call; cached locally afterwards
    project = paper.get_project("stack")
    results  = project.list("stack")

For papers with **multiple datasets** (e.g. two volcanoes), pass the
``project=`` keyword argument:

.. code-block:: python

    paper = mrp.get_paper("2023_Yates_PitonRuapehu")
    project_pdf     = paper.get_project("dvv", project="pdf")
    project_ruapehu = paper.get_project("dvv", project="ruapehu")

Cache management
-----------------

Downloaded archives are stored in the platform user-cache directory
(``~/.cache/msnoise-mrp/`` on Linux).  To free space:

.. code-block:: python

    mrp.clear_cache("2016_DePlaen_PitonDeLaFournaise")  # one paper
    mrp.clear_cache()                                    # all archives

Registry metadata (``registry.yaml``) and small paper files are never
deleted by ``clear_cache``.  To force a fresh registry download:

.. code-block:: python

    mrp = MRP(force_refresh=True)

Contributing a paper
---------------------

See the
`CONTRIBUTING guide <https://github.com/ROBelgium/MSNoise_Reproducible_Papers/blob/main/CONTRIBUTING.md>`_
in the registry repository.  The short version:

1. Fork ``MSNoise_Reproducible_Papers``, create ``papers/<YYYY_Author_Title>/``
2. Add ``project.yaml``, ``citation.bib``, ``meta.yaml``, ``README.md``
3. Run ``python scripts/update_registry.py && python scripts/update_readme.py``
4. Open a PR — CI validates schemas and runs ``msnoise db init`` on every
   ``project*.yaml``

.. seealso::

   - :ref:`msnoise_project` — full API reference for ``MSNoiseProject``
   - :ref:`msnoise_papers` — full API reference for the MRP client
   - :ref:`msnoise_result` — reading results with ``MSNoiseResult``
   - :doc:`notebooks/nb_mrp_analysis` — worked example notebook
