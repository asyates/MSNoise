"""FDSN/EIDA waveform fetching and bulk SDS download for MSNoise.

This module handles fetching raw waveforms from FDSN web services or the
EIDA routing client, writing optional raw caches, and per-station error
handling.  It is the **only** place in MSNoise that makes network calls to
external data services.
"""

__all__ = [
    "build_client",
    "fetch_and_preprocess",
    "fetch_raw_waveforms",
    "fetch_waveforms_bulk",
    "is_remote_source",
    "parse_datasource_scheme",
    "get_auth",
    "mass_download",
    "FDSNConnectionError",
]


import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("msnoise.fdsn")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class FDSNConnectionError(OSError):
    """Raised when an FDSN request fails due to a connection-level error.

    Distinct from :class:`~obspy.clients.fdsn.header.FDSNNoDataException`
    (no matching data) and auth errors (wrong credentials).  Callers that
    cache client objects should catch this, invalidate the cache, rebuild
    the client, and retry.
    """


def _is_connection_error(exc) -> bool:
    """Return True for errors that indicate a stale or broken connection."""
    msg = str(exc)
    # requests-level
    try:
        import requests.exceptions as _re
        if isinstance(exc, (_re.ConnectionError, _re.Timeout,
                            _re.ChunkedEncodingError)):
            return True
    except ImportError:
        pass
    # HTTP 503 / 504 from FDSN server
    if any(code in msg for code in ("503", "504", "timed out", "reset", "broken pipe",
                                     "RemoteDisconnected", "IncompleteRead")):
        return True
    return False


# ---------------------------------------------------------------------------
# URI scheme detection
# ---------------------------------------------------------------------------

def parse_datasource_scheme(uri: str) -> str:
    """Return the scheme of a DataSource URI.

    :param uri: DataSource.uri string.
    :returns: One of ``"local"``, ``"sds"``, ``"fdsn"``, ``"eida"``.
    """
    if not uri:
        return "local"
    from urllib.parse import urlsplit
    scheme = urlsplit(uri).scheme.lower()
    if scheme in ("", "sds"):
        return "sds" if scheme == "sds" else "local"
    if scheme == "fdsn":
        return "fdsn"
    if scheme == "eida":
        return "eida"
    return "local"


def is_remote_source(uri: str) -> bool:
    """Return True if the DataSource URI points to a remote service."""
    return parse_datasource_scheme(uri) in ("fdsn", "eida")


# ---------------------------------------------------------------------------
# Auth resolution
# ---------------------------------------------------------------------------

def get_auth(auth_env: str) -> dict:
    """Read credentials from environment variables for the given prefix.

    Looks up:
    - ``{auth_env}_FDSN_USER``
    - ``{auth_env}_FDSN_PASSWORD``
    - ``{auth_env}_FDSN_TOKEN``  (path to EIDA token file, or token string)

    :param auth_env: Env var prefix (e.g. ``"MSNOISE"``, ``"IRIS"``).
    :returns: Dict with keys ``user``, ``password``, ``token`` (any may be None).
    """
    prefix = auth_env.upper()
    return {
        "user":     os.environ.get(f"{prefix}_FDSN_USER"),
        "password": os.environ.get(f"{prefix}_FDSN_PASSWORD"),
        "token":    os.environ.get(f"{prefix}_FDSN_TOKEN"),
    }


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def build_client(ds):
    """Build an ObsPy client for the given DataSource.

    :param ds: :class:`~msnoise.msnoise_table_def.DataSource` ORM object.
    :returns: An ObsPy client with a ``get_waveforms_bulk`` method.
    """
    scheme = parse_datasource_scheme(ds.uri)
    auth = get_auth(ds.auth_env or "MSNOISE")

    if scheme == "fdsn":
        from obspy.clients.fdsn import Client
        # Strip "fdsn://" prefix — handle both fdsn://http://... and fdsn:///http://...
        raw = ds.uri[len("fdsn://"):]
        base_url = raw.lstrip("/")  # remove any leading slashes from triple-slash forms
        kwargs = {}
        if auth["user"] and auth["password"]:
            kwargs["user"] = auth["user"]
            kwargs["password"] = auth["password"]
        if auth["token"]:
            kwargs["eida_token"] = auth["token"]
        return Client(base_url, **kwargs)

    if scheme == "eida":
        from obspy.clients.fdsn import RoutingClient
        raw = ds.uri[len("eida://"):]
        base_url = raw.lstrip("/")
        kwargs = {}
        if auth["token"]:
            kwargs["eida_token"] = auth["token"]
        return RoutingClient(base_url, **kwargs)

    raise ValueError(f"build_client called for non-remote DataSource: {ds.uri!r}")


# ---------------------------------------------------------------------------
# Bulk fetch with retry + per-station error handling
# ---------------------------------------------------------------------------

def fetch_waveforms_bulk(client, bulk_request, retries=3):
    """Issue a bulk waveform request with retry logic.

    :param client: ObsPy FDSN/EIDA client.
    :param bulk_request: List of ``(net, sta, loc, chan, t1, t2)`` tuples.
    :param retries: Number of retry attempts for transient errors.
    :returns: :class:`~obspy.core.stream.Stream` (may be empty on failure).
    :raises: Re-raises auth errors and no-data exceptions immediately.
    """
    from obspy.clients.fdsn.header import FDSNNoDataException
    from obspy import Stream

    last_exc = None
    last_is_conn = False
    for attempt in range(1, retries + 1):
        try:
            return client.get_waveforms_bulk(bulk_request)
        except FDSNNoDataException:
            raise  # not transient — re-raise immediately
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # Auth failures are not transient
            if "401" in err_str or "403" in err_str or "Unauthorized" in err_str:
                raise
            last_is_conn = _is_connection_error(exc)
            wait = 2 ** attempt
            logger.warning(
                f"FDSN bulk request failed (attempt {attempt}/{retries}): "
                f"{exc}. Retrying in {wait}s."
            )
            time.sleep(wait)

    if last_is_conn:
        raise FDSNConnectionError(
            f"FDSN connection lost after {retries} attempts: {last_exc}"
        ) from last_exc
    logger.error(f"FDSN bulk request failed after {retries} attempts: {last_exc}")
    return Stream()


# ---------------------------------------------------------------------------
# High-level: fetch + optional raw cache + per-station result
# ---------------------------------------------------------------------------

def fetch_raw_waveforms(db, jobs, goal_day, params, t_start=None, t_end=None,
                        client=None):
    """Fetch raw (unprocessed) waveforms from FDSN/EIDA for a batch of stations.

    Issues one ``get_waveforms_bulk`` call covering all stations in *jobs*,
    optionally writes a raw file cache (``fdsn_keep_raw=Y``), and returns
    the combined :class:`~obspy.core.stream.Stream`.  No preprocessing is
    applied — the caller receives exactly what the FDSN server returns.

    Use this when downstream processing needs raw data (e.g. PSD computation
    via ObsPy PPSD, which handles its own response correction internally).
    For pre-processing + response removal use :func:`fetch_and_preprocess`.

    :param db: SQLAlchemy session.
    :param jobs: List of Job ORM objects (all same DataSource).
    :param goal_day: Date string ``YYYY-MM-DD``.
    :param params: :class:`~msnoise.params.MSNoiseParams` for this lineage.
    :param t_start: Optional :class:`~obspy.core.utcdatetime.UTCDateTime`
        override for the fetch window start (default: midnight of *goal_day*).
    :param t_end: Optional :class:`~obspy.core.utcdatetime.UTCDateTime`
        override for the fetch window end (default: midnight + 86400 s).
    :returns: :class:`~obspy.core.stream.Stream` (may be empty on failure).
    """
    from obspy import UTCDateTime, Stream
    from obspy.clients.fdsn.header import FDSNNoDataException
    from .stations import resolve_data_source, get_station

    log = logging.getLogger("msnoise.fdsn.fetch_raw")

    first_job = jobs[0]
    net0, sta0, _ = first_job.pair.split(".")
    ds = resolve_data_source(db, get_station(db, net0, sta0))

    fdsn_keep_raw = params.global_.fdsn_keep_raw
    retries       = int(params.global_.fdsn_retries or 3)
    output_folder = params.global_.output_folder
    step_name     = getattr(params, "step_name", goal_day)

    t1 = t_start if t_start is not None else UTCDateTime(goal_day)
    t2 = t_end   if t_end   is not None else t1 + 86400

    # Build bulk request from station channel info
    bulk = []
    for job in jobs:
        net, sta, loc = job.pair.split(".")
        fdsn_loc = "" if loc == "--" else loc
        station_obj = get_station(db, net, sta)
        chans = station_obj.chans() if station_obj and station_obj.chans() else []
        if chans:
            for chan in chans:
                bulk.append((net, sta, fdsn_loc, chan, t1, t2))
        else:
            # Fallback: wildcard per station
            bulk.append((net, sta, fdsn_loc, "*", t1, t2))

    log.info(
        f"FDSN raw fetch: {len(jobs)} station(s), {len(bulk)} channel(s), "
        f"day={goal_day}, source={ds.name!r}, window={t1}–{t2}"
    )

    try:
        if client is None:
            client = build_client(ds)
        raw_stream = fetch_waveforms_bulk(client, bulk, retries=retries)
    except FDSNConnectionError:
        raise  # let caller invalidate the client cache and rebuild
    except FDSNNoDataException:
        log.warning(f"No data from {ds.name!r} for {goal_day}")
        return Stream()
    except Exception as exc:
        if "401" in str(exc) or "403" in str(exc) or "Unauthorized" in str(exc):
            log.error(
                f"Auth failure for DataSource {ds.name!r}. "
                f"Check {ds.auth_env}_FDSN_USER / {ds.auth_env}_FDSN_TOKEN. "
                f"Error: {exc}"
            )
        else:
            log.error(f"FDSN raw fetch failed for {ds.name!r} on {goal_day}: {exc}")
        return Stream()

    if fdsn_keep_raw in ("Y", "y", True):
        _write_raw_cache(raw_stream, output_folder, step_name, goal_day)

    return raw_stream


def fetch_and_preprocess(
    db, jobs, goal_day, params, responses=None, loglevel="INFO", client=None
):
    """Fetch waveforms from FDSN/EIDA for a batch of stations on one day.

    Groups jobs by DataSource (all jobs in a batch share the same
    ``data_source_id`` when ``group_by="day_lineage_datasource"`` is used),
    issues one ``get_waveforms_bulk`` call, splits the result by station,
    and optionally writes raw files.

    :param db: SQLAlchemy session.
    :param jobs: List of Job ORM objects (all same day, same DataSource).
    :param goal_day: Date string ``YYYY-MM-DD``.
    :param params: :class:`~msnoise.params.MSNoiseParams` for this lineage.
    :param responses: ObsPy Inventory for instrument response removal, or None.
    :param loglevel: Logging level string.
    :returns: Tuple ``(stream, done_jobs, failed_jobs)`` where stream contains
        preprocessed traces for all successfully fetched stations.
    """
    from obspy import UTCDateTime, Stream
    from .preprocessing import apply_preprocessing_to_stream

    log = logging.getLogger("msnoise.fdsn.fetch")

    min_coverage = float(params.global_.fdsn_min_coverage or 0.5)
    t1 = UTCDateTime(goal_day)
    t2 = t1 + 86400

    # Delegate the actual FDSN fetch (+ optional raw cache) to fetch_raw_waveforms
    raw_stream = fetch_raw_waveforms(db, jobs, goal_day, params, t_start=t1, t_end=t2,
                                     client=client)

    if not raw_stream:
        return Stream(), [], jobs

    # Build station_map for splitting the combined stream per job
    station_map = {job.pair: job for job in jobs}

    # Split by station, check coverage, apply preprocessing
    done_jobs, failed_jobs = [], []
    out_stream = Stream()

    for sid, job in station_map.items():
        net, sta, loc = sid.split(".")
        fdsn_loc = "" if loc == "--" else loc
        sta_stream = raw_stream.select(network=net, station=sta, location=fdsn_loc)
        # Normalise "" → "--" in the fetched stream to match MSNoise convention
        for tr in sta_stream:
            if tr.stats.location == "":
                tr.stats.location = "--"

        if not sta_stream:
            log.warning(f"Station {sid} absent from FDSN bulk response — marking Failed")
            failed_jobs.append(job)
            continue

        # Coverage check
        total = sum(tr.stats.delta * tr.stats.npts for tr in sta_stream)
        if total < min_coverage * 86400:
            log.warning(
                f"{sid}: coverage {total/86400:.1%} < {min_coverage:.0%} — "
                f"proceeding with available data"
            )

        # Apply the full preprocessing pipeline (taper, filter, resample,
        # optional response removal) — same as the local-archive path.
        processed = apply_preprocessing_to_stream(
            sta_stream, params, responses=responses, logger=log
        )
        if processed:
            out_stream += processed
            done_jobs.append(job)
        else:
            log.warning(f"{sid}: preprocessing produced empty stream — marking Failed")
            failed_jobs.append(job)

    return out_stream, done_jobs, failed_jobs


def _write_raw_cache(stream, output_folder, step_name, goal_day):
    """Write raw fetched stream to ``_output/raw/<date>/<NET.STA.LOC.CHAN>.mseed``."""
    from obspy import Stream as _Stream

    raw_dir = os.path.join(output_folder, step_name, "_output", "raw", goal_day)
    os.makedirs(raw_dir, exist_ok=True)

    # Group by full SEED ID (NET.STA.LOC.CHAN)
    by_id = {}
    for tr in stream:
        sid = tr.id  # NET.STA.LOC.CHAN
        by_id.setdefault(sid, _Stream())
        by_id[sid].append(tr)

    for sid, st in by_id.items():
        fpath = os.path.join(raw_dir, f"{sid}.mseed")
        for tr in st:
            tr.data = tr.data.astype("float32")
        st.write(fpath, format="MSEED")
        logger.debug(f"Raw cache written: {fpath}")


# ---------------------------------------------------------------------------
# MassDownloader — bulk SDS population
# ---------------------------------------------------------------------------

_DEFAULT_SDS = Path("SDS")


def _resolve_sds_root(session, schema, sds_root: Path | None) -> Path:
    """Return the SDS write root, with fallback heuristic and logging.

    Resolution order:

    1. Explicit *sds_root* argument (from ``--sds-path`` CLI option).
    2. Exactly one local SDS DataSource (``data_structure="SDS"``, non-remote
       URI) → use its URI.
    3. Fall back to ``./SDS`` with a warning.

    :param session: SQLAlchemy session.
    :param schema: Declared schema object.
    :param sds_root: Explicit path from caller, or ``None``.
    :returns: Resolved :class:`~pathlib.Path`.
    """
    if sds_root is not None:
        logger.info("SDS root (explicit): %s", sds_root)
        return sds_root

    all_sources = session.query(schema.DataSource).all()
    local_sds = [
        ds for ds in all_sources
        if not is_remote_source(ds.uri) and ds.uri
        and ds.data_structure == "SDS"
    ]

    if len(local_sds) == 1:
        resolved = Path(local_sds[0].uri)
        logger.info("SDS root (from DataSource %r): %s", local_sds[0].name, resolved)
        return resolved

    if len(local_sds) > 1:
        logger.warning(
            "Multiple local SDS DataSources found (%s) — cannot pick one "
            "automatically.  Defaulting to %s.  Use --sds-path to override.",
            ", ".join(repr(ds.name) for ds in local_sds),
            _DEFAULT_SDS,
        )
    else:
        logger.warning(
            "No local SDS DataSource found.  Defaulting to %s.  "
            "Use --sds-path to override.",
            _DEFAULT_SDS,
        )
    return _DEFAULT_SDS


def _sds_mseed_storage(sds_root: Path):
    """``mseed_storage`` callable: route traces into day-aligned SDS files.

    Returns ``True`` (skip) for files that already exist.

    :param sds_root: Root of the SDS archive.
    """
    def fn(network, station, location, channel, starttime, _endtime):
        loc = "" if (location is None or location == "--") else location
        fname = (
            f"{network}.{station}.{loc}.{channel}.D"
            f".{starttime.year}.{starttime.julday:03d}"
        )
        dest = (
            sds_root
            / str(starttime.year)
            / network
            / station
            / f"{channel}.D"
            / fname
        )
        if dest.exists():
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        return str(dest)
    return fn


def _stationxml_storage(xml_root: Path):
    """``stationxml_storage`` callable: persist StationXML, never overwrite.

    :param xml_root: Directory for StationXML files.
    """
    xml_root.mkdir(parents=True, exist_ok=True)

    def fn(network, station):
        dest = xml_root / f"{network}.{station}.xml"
        return True if dest.exists() else str(dest)
    return fn


def mass_download(session, schema, sds_root=None,
                  startdate_override=None, enddate_override=None):
    """Download waveforms for all remote DataSources into an SDS archive.

    Builds an ObsPy :class:`~obspy.core.inventory.Inventory` from the station
    table and passes it as ``limit_stations_to_inventory`` — the
    MassDownloader gates at station level server-side, avoiding any NSLC
    cartesian blowup.  Channel codes are the union of all ``Station.chans()``.

    ``chunklength_in_sec=86400`` produces day-aligned SDS files.  StationXML
    is stored under ``<sds_root>/../stationxml/`` and never overwritten.
    ``sanitize=False`` prevents discarding traces for missing response.

    :param session: SQLAlchemy session.
    :param schema: Declared schema object (from ``declare_tables()``).
    :param sds_root: SDS write root as :class:`~pathlib.Path`, or ``None``
        to auto-resolve (see :func:`_resolve_sds_root`).
    :param startdate_override: Override project startdate (``YYYY-MM-DD``).
    :param enddate_override: Override project enddate (``YYYY-MM-DD``).
    :raises ValueError: If no remote DataSource or no valid stations found.
    """
    from collections import defaultdict

    from obspy import UTCDateTime
    from obspy.core.inventory import Inventory, Network
    from obspy.core.inventory import Station as OBStation
    from obspy.clients.fdsn.mass_downloader import (
        GlobalDomain, MassDownloader, Restrictions,
    )
    from .config import get_config

    # --- date range ---
    startdate = startdate_override or get_config(
        session, "startdate", category="global", set_number=1)
    enddate = enddate_override or get_config(
        session, "enddate", category="global", set_number=1)
    t_start = UTCDateTime(startdate)
    t_end   = UTCDateTime(enddate) + 86400

    # --- remote providers ---
    all_sources    = session.query(schema.DataSource).all()
    remote_sources = [ds for ds in all_sources if is_remote_source(ds.uri)]
    if not remote_sources:
        raise ValueError(
            "No remote (FDSN/EIDA) DataSource found.  "
            "`msnoise utils download` requires at least one fdsn:// or eida:// source."
        )

    # --- SDS + StationXML paths ---
    sds_root = _resolve_sds_root(session, schema, sds_root)
    xml_root = sds_root.parent / "stationxml"
    logger.info("StationXML: %s", xml_root)

    # --- inventory from station table ---
    stations = (
        session.query(schema.Station)
        .filter(schema.Station.used == True)  # noqa: E712
        .all()
    )
    nets: dict = defaultdict(list)
    for sta in stations:
        locs  = sta.locs()
        chans = sta.chans()
        if not locs:
            logger.warning("%s.%s has no used_location_codes — skipping.", sta.net, sta.sta)
            continue
        if not chans:
            logger.warning("%s.%s has no used_channel_names — skipping.", sta.net, sta.sta)
            continue
        nets[sta.net].append(OBStation(
            code=sta.sta,
            latitude=sta.Y or 0.0,
            longitude=sta.X or 0.0,
            elevation=sta.altitude or 0.0,
            creation_date=UTCDateTime(0),
        ))
    if not nets:
        raise ValueError(
            "No valid stations found.  Run `msnoise populate` before downloading."
        )
    inv = Inventory(
        networks=[Network(code=net, stations=stas) for net, stas in nets.items()],
        source="MSNoise",
    )
    n_sta = sum(len(s) for s in nets.values())
    logger.info("Inventory: %d network(s), %d station(s).", len(nets), n_sta)

    all_chans = ",".join(sorted({c for sta in stations for c in sta.chans()}))
    logger.info("Channel union: %s", all_chans)

    # --- single MassDownloader call ---
    restr = Restrictions(
        starttime=t_start,
        endtime=t_end,
        network="*",
        station="*",
        channel=all_chans,
        limit_stations_to_inventory=inv,
        chunklength_in_sec=86400,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        sanitize=False,
    )
    mdl = MassDownloader(providers=[build_client(ds) for ds in remote_sources])
    logger.info("Starting MassDownload: %s → %s, %d source(s), SDS: %s",
                startdate, enddate, len(remote_sources), sds_root)
    mdl.download(
        GlobalDomain(), restr,
        mseed_storage=_sds_mseed_storage(sds_root),
        stationxml_storage=_stationxml_storage(xml_root),
    )
    logger.info("MassDownload complete.")
