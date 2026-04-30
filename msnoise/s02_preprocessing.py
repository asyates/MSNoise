"""
This step preprocesses waveforms using the preprocessing.py module and saves
the resulting Stream objects to disk in a workflow-aware folder structure.

For local/SDS sources, waveforms are read from the archive via
DataAvailability records.  For FDSN/EIDA sources, waveforms are fetched from
the remote service using ``get_waveforms_bulk`` and optionally cached as raw
files (``fdsn_keep_raw=Y``) before preprocessing.

Configuration Parameters
------------------------

* |preprocess.preprocess_components|
* |preprocess.remove_response|
* |global.response_path|
* |global.fdsn_keep_raw|
* |global.hpc|

.. automodule:: msnoise.core.preprocessing

"""

import time
import traceback
import numpy as np
from .core.db import connect, get_logger
from .core.workflow import get_next_lineage_batch, is_next_job_for_step, massive_update_job
from .core.signal import preload_instrument_responses, save_preprocessed_streams
from .core.stations import resolve_data_source, get_station
from .core.fdsn import is_remote_source, fetch_and_preprocess, build_client, FDSNConnectionError
from .core.preprocessing import preprocess

CATEGORY = "preprocess"


def main(loglevel="INFO"):
    """
    Main preprocessing workflow function.

    Dispatches to the local-archive or FDSN/EIDA fetch path depending on
    the station's DataSource URI scheme, then preprocesses and writes
    per-station output files.
    """
    logger = get_logger(f"msnoise.{CATEGORY}", loglevel, with_pid=True)
    logger.info('*** Starting: Preprocessing Step ***')

    db = connect()

    # Cache FDSN clients keyed by DataSource.ref — built lazily, reused across days.
    # Avoids re-opening HTTP connections for every day's batch.
    _client_cache = {}

    while is_next_job_for_step(db, step_category=CATEGORY):
        batch = get_next_lineage_batch(db, step_category=CATEGORY,
                                       group_by="day_lineage", loglevel=loglevel)
        if batch is None:
            time.sleep(np.random.random())
            continue

        jobs       = batch["jobs"]
        step       = batch["step"]
        params     = batch["params"]
        days       = batch["days"]

        goal_day   = days[0]
        step_name  = step.step_name
        output_dir = params.global_.output_folder

        logger.info(f"Processing {len(jobs)} jobs for step '{step_name}' on {goal_day}")

        # Mark all in-progress atomically before processing
        massive_update_job(db, jobs, "I")

        try:
            raw = params.preprocess.preprocess_components
            components = raw.split(',') if isinstance(raw, str) else (list(raw) if raw else ['Z'])

            if params.preprocess.remove_response in ('Y', 'y', True):
                logger.debug('Pre-loading all instrument responses')
                responses = preload_instrument_responses(db, return_format="inventory")
            else:
                responses = None

            # ── Resolve DataSource per station — group by (ds_ref, remote) ──
            # A batch may contain stations from different DataSources (e.g.
            # some from a local SDS archive, others from an FDSN service).
            # Resolving only from the first job was wrong for mixed batches.
            from collections import defaultdict
            _jobs_by_ds = defaultdict(list)
            for _job in jobs:
                _net, _sta, _ = _job.pair.split(".")
                _sta_obj = get_station(db, _net, _sta)
                _ds = resolve_data_source(db, _sta_obj)
                _jobs_by_ds[(_ds.ref, is_remote_source(_ds.uri), _ds)].append(_job)

            from obspy import Stream as _Stream
            stream     = _Stream()
            done_jobs  = []
            failed_jobs = []

            for (_ds_ref, _remote, _ds), _ds_jobs in _jobs_by_ds.items():
                if _remote:
                    # ── FDSN / EIDA path ─────────────────────────────────────
                    logger.info(
                        f"DataSource {_ds.name!r} is remote ({_ds.uri}) — "
                        f"fetching {len(_ds_jobs)} station(s) via FDSN/EIDA"
                    )
                    if _ds.ref not in _client_cache:
                        logger.debug(f"Building FDSN client for DataSource {_ds.name!r}")
                        _client_cache[_ds.ref] = build_client(_ds)
                    for _attempt in range(2):
                        try:
                            _st, _done, _failed = fetch_and_preprocess(
                                db, _ds_jobs, goal_day, params,
                                responses=responses, loglevel=loglevel,
                                client=_client_cache[_ds.ref],
                            )
                            break
                        except FDSNConnectionError as _e:
                            if _attempt == 0:
                                logger.warning(
                                    f"FDSN connection lost for {_ds.name!r} "
                                    f"({_e}), rebuilding client and retrying."
                                )
                                _client_cache[_ds.ref] = build_client(_ds)
                            else:
                                logger.error(
                                    f"FDSN connection failed again after rebuild "
                                    f"for {_ds.name!r} on {goal_day} — marking jobs Failed."
                                )
                                _st, _done, _failed = _Stream(), [], _ds_jobs
                    stream      += _st
                    done_jobs   += _done
                    failed_jobs += _failed
                else:
                    # ── Local / SDS path ─────────────────────────────────────
                    _stations = [j.pair for j in _ds_jobs]
                    logger.debug(f"Processing stations (local, DataSource {_ds.name!r}): {_stations}")
                    _st = preprocess(_stations, components, goal_day, params,
                                     responses=responses, loglevel=loglevel)
                    _ids = {f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}"
                            for tr in _st}
                    done_jobs   += [j for j in _ds_jobs if j.pair in _ids]
                    failed_jobs += [j for j in _ds_jobs if j.pair not in _ids]
                    stream += _st

            # ── Write per-station output files ───────────────────────────────
            if stream:
                saved_files = save_preprocessed_streams(
                    stream, output_dir, step_name, goal_day)
                logger.info(f"Saved {len(saved_files)} preprocessed file(s)")

            if done_jobs:
                massive_update_job(db, done_jobs, "D")
            if failed_jobs:
                logger.warning(
                    f"{len(failed_jobs)} job(s) for {step_name} on {goal_day} "
                    f"marked Failed (station not in stream)"
                )
                massive_update_job(db, failed_jobs, "F")

            if not batch["params"].global_.hpc:
                from .core.workflow import propagate_downstream
                propagate_downstream(db, batch)

        except Exception:
            logger.error(f"Error processing step {step_name} on {goal_day}:")
            logger.error(traceback.format_exc())
            massive_update_job(db, jobs, "F")

    logger.info('*** Finished: Preprocessing Step ***')


if __name__ == "__main__":
    main()
