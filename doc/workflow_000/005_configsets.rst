.. include:: ../configs.hrst

.. _workflow_configsets:

Setting up a Multi-Branch Project
==================================

This page walks through configuring a project that uses **multiple config sets**
to run parallel processing branches — for example, two filter frequency bands
or two MWCS window lengths.

For the conceptual background, see :ref:`workflow_concepts`.

.. contents::
    :local:


Default project (one branch)
------------------------------

A freshly initialised project has one config set per category:
``cc_1``, ``filter_1``, ``stack_1``, ``refstack_1``, ``mwcs_1``, etc.
The whole pipeline runs as a single linear chain.


Adding a second filter band
-----------------------------

1. Create the second config set and edit its parameters:

   .. code-block:: sh

       msnoise config create_set filter
       msnoise config list_sets                 # should show filter_1 and filter_2
       msnoise config set filter.2.freqmin 1.0
       msnoise config set filter.2.freqmax 5.0

2. Apply the new topology to the database:

   .. code-block:: sh

       msnoise db upgrade

   This creates ``filter_2`` as a new WorkflowStep and wires it:
   ``cc_1 → filter_2 → stack_1`` and ``cc_1 → filter_2 → refstack_1`` (siblings)

3. Run the CC step as usual.  Both filter branches are processed:

   .. code-block:: sh

       msnoise cc compute
       msnoise cc stack
       msnoise cc stack_refstack
       msnoise cc dtt compute_mwcs
       …

   Output files for ``filter_1`` land under::

       OUTPUT/preprocess_1/cc_1/filter_1/stack_1/…

   Output files for ``filter_2`` land under::

       OUTPUT/preprocess_1/cc_1/filter_2/stack_1/…


Adding a second MWCS parameterisation
---------------------------------------

.. code-block:: sh

    msnoise config create_set mwcs
    msnoise config copy_set mwcs 1 mwcs 2    # start from set 1's values
    msnoise config set mwcs.2.freqmin 1.0
    msnoise config set mwcs.2.freqmax 5.0
    msnoise config set mwcs.2.mwcs_wlen 5
    msnoise db upgrade

Both ``mwcs_1`` and ``mwcs_2`` will run on every refstack branch.


Checking what was created
--------------------------

.. code-block:: sh

    msnoise config list_sets          # all categories and set numbers
    msnoise admin                     # web UI → Workflow → Steps

From Python:

.. code-block:: python

    from msnoise.plugins import connect, get_workflow_steps
    db = connect()
    for s in get_workflow_steps(db):
        print(f"{s.step_name:25s}  cat={s.category}  set={s.set_number}")


Resetting jobs after a config change
--------------------------------------

If you add a new config set after data has already been processed, reset
the relevant steps so the new branch gets processed:

.. code-block:: sh

    msnoise reset stack_1 --all      # reset all stack jobs to T
    msnoise new_jobs --after cc      # re-propagate from cc → stack → …

.. _workflow_project_yaml:

Initialising from a project YAML
----------------------------------

For reproducible or paper-derived setups, the entire config topology — config
sets, workflow links, data source, and stations — can be expressed in a single
**project YAML** file and applied in one command::

    msnoise db init --from-yaml myproject.yaml

The project YAML uses ``category_N`` keys (not bare category names) so that
multiple config sets of the same category are unambiguous.  Each entry may
declare an ``after`` field (string or list) that maps directly to
``WorkflowLink`` rows — no ALL×ALL fan-out, no manual link editing.

Minimal example::

    msnoise_project_version: 1

    global_1:
      startdate: "2013-04-01"
      enddate:   "2014-10-31"

    preprocess_1:
      after: global_1
      cc_sampling_rate: 20.0

    cc_1:
      after: preprocess_1
      cc_type_single_station_AC: PCC
      whitening: "N"

    filter_1:
      after: cc_1
      freqmin: 1.0
      freqmax: 2.0
      AC: "Y"

    filter_2:
      after: cc_1          # fan-out: both filters fed from the same cc step
      freqmin: 0.5
      freqmax: 1.0
      AC: "Y"

    stack_1:
      after: [filter_1, filter_2]
      mov_stack: "(('2D','1D'))"

    refstack_1:
      after: [filter_1, filter_2]   # sibling of stack, not child
      ref_begin: "2013-04-01"
      ref_end:   "2014-10-31"

    mwcs_1:
      after: [stack_1, refstack_1]  # join condition: requires both parents
      freqmin: 1.0
      freqmax: 2.0

    mwcs_dtt_1:
      after: mwcs_1
      dtt_minlag: 5.0
      dtt_width:  30.0

Only overrides need to be listed — all other parameters keep their CSV defaults.
Unknown keys produce a warning (typo guard).

Stations and data sources can be declared in the same file::

    data_sources:
      - name: EPOS-FR
        uri: "fdsn://http://ws.resif.fr"
        data_structure: SDS
        auth_env: MSNOISE

    stations:
      # Fetch all stations with full response (saved to response_path automatically):
      station_endpoint: "http://ws.resif.fr/fdsnws/station/1/query?network=YA&level=response&starttime=2013-04-01&endtime=2014-10-31"

Or, when data are not on a public FDSN service, list stations explicitly
(same format as the provenance block exported by ``MSNoiseResult``)::

    data_sources:
      - name: local
        uri: ""
        data_structure: SDS
        auth_env: MSNOISE

    stations:
      - net: IV
        sta: ECPN
        X: 14.9905
        Y: 37.7480
        altitude: 2700.0
        coordinates: DEG
        data_source_id: 0

.. note::

   Two YAML schemas exist — do not confuse them:

   * **Per-lineage params** (``msnoise_params_version: 1``, bare ``filter:`` keys):
     written by :meth:`~msnoise.params.MSNoiseParams.to_yaml`, read by
     :meth:`~msnoise.params.MSNoiseParams.from_yaml` and
     :class:`~msnoise.results.MSNoiseResult`.  Represents one pipeline path.

   * **Project seed** (``msnoise_project_version: 1``, ``filter_1:`` keys):
     hand-written or generated from a paper template, consumed by
     ``msnoise db init --from-yaml``.  Represents a full project topology.

See :func:`~msnoise.core.config.create_project_from_yaml` for the full API.
