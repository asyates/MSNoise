.. include:: ../configs.hrst

.. _fdsn_download:

Bulk Waveform Download (FDSN/EIDA)
===================================

When your project uses a remote FDSN or EIDA :class:`DataSource`, there are
two ways to get waveforms into the pipeline:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - How it works
     - When to use
   * - **Bulk download**
     - ``msnoise utils download`` fetches all waveforms up-front into a local
       SDS archive using ObsPy's MassDownloader.  Subsequent steps read from
       disk — no network access during processing.
     - Large date ranges, HPC clusters, reproducible paper datasets, slow or
       unreliable FDSN connections.
   * - **On-the-fly (stream)**
     - The preprocess step fetches each day's waveforms directly from FDSN
       during processing.  No local archive needed.
     - Short date ranges, fast reliable connections, interactive exploration.


Bulk download workflow
-----------------------

.. code-block:: sh

    # 1. Initialise the project (from a YAML or manually)
    msnoise db init --from-yaml project.yaml

    # 2. Populate station metadata (sets used_location_codes / used_channel_names)
    msnoise utils import-stationxml https://...
    # or: msnoise admin → Stations → mark used=True

    # 3. Download all raw waveforms into the local SDS archive
    msnoise utils download

    # 4. Scan the archive to populate data availability
    msnoise scan_archive --init

    # 5. Run the full pipeline
    msnoise utils run_workflow

The SDS write root is resolved automatically in this order:

1. ``--sds-path PATH`` option (explicit override).
2. The single unambiguous local SDS :class:`DataSource` URI from the database.
3. ``./SDS`` (with a warning when no local SDS DataSource is found, or when
   multiple local SDS sources are configured).

StationXML files are stored under ``<sds_root>/../stationxml/`` and are never
overwritten if already present.  Traces are never discarded for missing
instrument response (``sanitize=False``).

.. code-block:: sh

    # Override the SDS path or restrict the date range:
    msnoise utils download --sds-path /data/SDS
    msnoise utils download --startdate 2013-06-01 --enddate 2013-06-30


On-the-fly (stream) workflow
------------------------------

.. code-block:: sh

    # 1. Initialise the project
    msnoise db init --from-yaml project.yaml

    # 2. Create preprocess jobs directly (no scan_archive needed)
    msnoise utils create_preprocess_jobs --date_range 2013-04-01 2014-10-31

    # 3. Run the pipeline — preprocess fetches from FDSN on demand
    msnoise utils run_workflow


Prompt at ``db init``
----------------------

When ``msnoise db init --from-yaml`` detects a remote DataSource **and** the
project has explicit ``startdate``/``enddate`` values, it prompts:

.. code-block:: text

    DataSource(s) 'resif-fdsn' use remote FDSN/EIDA.
      How would you like to proceed?
      [1] Bulk download first  — fetch all raw waveforms into SDS, then run pipeline
      [2] Stream from FDSN     — preprocess fetches on-the-fly (creates jobs now)
      [3] Skip                 — I'll handle it manually

