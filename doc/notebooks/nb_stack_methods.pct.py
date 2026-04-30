# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Stack Method Comparison — linear / pws / tf-PWS Reference Stacks

This notebook builds a minimal MSNoise project and compares three stacking
methods for the reference (REF) cross-correlation functions:

| Refstack    | ``stack_method`` | Reference                          |
|-------------|------------------|------------------------------------|
| `refstack_1`| `linear`         | arithmetic mean                    |
| `refstack_2`| `pws`            | Schimmel & Paulssen (1997)         |
| `refstack_3`| `tfpws`          | Schimmel & Gallart (2007)          |

Both **inter-station CC** (classical cross-correlation between station pairs,
ZZ component) and **single-station AC** (autocorrelation of each station's Z
component, using PCC to suppress the zero-lag spike) are computed through a
single shared `cc_1` step.

The final figures compare the three REF stacks:

1. **Time domain** — normalised waveforms overlaid (one panel per pair)
2. **Frequency domain** — amplitude spectrum (one panel per pair)

## Prerequisites

* MSNoise installed with tf-PWS support.
* A one-day MiniSEED dataset for at least two stations.
  The classic MSNoise tutorial dataset (network `YA`, stations `UV05`,
  `UV06`, `UV10`, day 2010-09-01, PDF archive layout) is used by default.
"""

# %% [markdown]
# ## 0 · Imports

# %%
# %matplotlib inline
import os
import logging
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from msnoise.core.db import connect, create_database_inifile
from msnoise.core.config import create_config_set, update_config
from msnoise.core.stations import update_station
from msnoise.core.workflow import (
    create_workflow_steps_from_config_sets,
    create_workflow_links_from_steps,
    get_workflow_steps,
)
from msnoise.msnoise_table_def import declare_tables, DataAvailability
from msnoise.results import MSNoiseResult

from msnoise.s01_scan_archive import main as scan_archive
from msnoise.s02_new_jobs import main as new_jobs
from msnoise.s02_preprocessing import main as preprocess
from msnoise.s03_compute_no_rotation import main as compute_cc
from msnoise.s04_stack_refstack import main as stack_refstack

logging.basicConfig(level=logging.WARNING)

# %% [markdown]
# ## 1 · User Settings

# %%
# ── Waveform archive ─────────────────────────────────────────────────────────
DATA_PATH      = r"C:\Users\tlecocq\AppData\Local\msnoise-testdata\msnoise-testdata\Cache\1.1\classic\data"
DATA_STRUCTURE = "PDF"          # "SDS" or "PDF"
NETWORK_CODE   = "YA"
CHANNELS       = "HHZ"

# ── Date range ───────────────────────────────────────────────────────────────
STARTDATE = "2010-09-01"
ENDDATE   = "2010-09-01"

# ── Stations (net, sta, lon°E, lat°N, elev_m) ────────────────────────────────
STATIONS = [
    ("YA", "UV05",  29.735,  -17.817, 1174.0),
    ("YA", "UV06",  29.785,  -17.827, 1162.0),
    ("YA", "UV10",  29.790,  -17.847, 1180.0),
]

# ── Signal processing ────────────────────────────────────────────────────────
CC_SAMPLING_RATE = 20.0   # Hz — resample target
MAXLAG           = 50.0   # s — short lag: captures AC reflections at 1–5 Hz
                           #     and near-station surface waves for CC pairs
CORR_DURATION    = 1800.0 # s — 30-min windows

# ── Filter ───────────────────────────────────────────────────────────────────
FREQMIN = 1.0   # Hz
FREQMAX = 5.0   # Hz

# ── PWS / tf-PWS parameters (shared by refstack_2 and refstack_3) ────────────
PWS_TIMEGATE   = 5.0   # s  — smoothing window for pws (< MAXLAG)
PWS_POWER      = 2.0   # exponent for coherence weight
TFPWS_NSCALES  = 20    # CWT scales spanning [FREQMIN, FREQMAX]

# ── Working directory ────────────────────────────────────────────────────────
WORK_DIR = None   # None → auto temp dir

# %% [markdown]
# ## 2 · Create Project Directory and Database

# %%
if WORK_DIR is None:
    WORK_DIR = tempfile.mkdtemp(prefix="msnoise_stackmethods_")
    print(f"Created temporary project directory: {WORK_DIR}")
else:
    os.makedirs(WORK_DIR, exist_ok=True)
    print(f"Using project directory: {WORK_DIR}")

os.chdir(WORK_DIR)

create_database_inifile(
    tech=1,
    hostname=os.path.join(WORK_DIR, "msnoise.sqlite"),
    database="", username="", password="", prefix="",
)

db = connect()
declare_tables().Base.metadata.create_all(db.get_bind())
print("Database created:", os.path.join(WORK_DIR, "msnoise.sqlite"))

# %% [markdown]
# ## 3 · Create Config Sets
#
# One configset per category.  For `refstack` we create **three** — one per
# `stack_method`.

# %%
SINGLE_CONFIGSET_CATEGORIES = [
    "global", "preprocess", "cc", "filter"
]

for cat in SINGLE_CONFIGSET_CATEGORIES:
    sn = create_config_set(db, cat)
    print(f"  {cat}_{sn} created")

# Three refstack configsets
ref_sn_linear = create_config_set(db, "refstack")
ref_sn_pws    = create_config_set(db, "refstack")
ref_sn_tfpws  = create_config_set(db, "refstack")

print(f"  refstack_{ref_sn_linear} ← linear")
print(f"  refstack_{ref_sn_pws}    ← pws")
print(f"  refstack_{ref_sn_tfpws}  ← tfpws")

db.commit()

# %% [markdown]
# ## 4 · Configure Global Parameters

# %%
OUTPUT_FOLDER = os.path.join(WORK_DIR, "OUTPUT")

for k, v in {
    "output_folder": OUTPUT_FOLDER,
    "startdate":     STARTDATE,
    "enddate":       ENDDATE,
    "hpc":           "N",
}.items():
    update_config(db, k, v, category="global", set_number=1)
    print(f"  global.{k} = {v!r}")

# %% [markdown]
# ## 5 · Configure CC Step
#
# A **single** `cc_1` step computes both:
#
# * Inter-station CC (`components_to_compute = "ZZ"`, `cc_type = "CC"`)
# * Single-station AC (`components_to_compute_single_station = "ZZ"`,
#   `cc_type_single_station_AC = "PCC"`)
#
# `whitening = "A"` (auto) applies spectral whitening to inter-station CC
# **but not** to AC — the AC branch in `s03` always starts from the
# bandpassed (`_data_bp`) copy of the data, never the whitened one.  This
# is enforced by design in the code (lines 866–903 of `s03_compute_no_rotation`).
#
# PCC for AC suppresses the zero-lag spike without whitening, making it the
# preferred AC algorithm for basin-depth imaging
# (Romero & Schimmel 2018; De Plaen et al. 2019).

# %%
cc_params = {
    "cc_sampling_rate":  str(CC_SAMPLING_RATE),
    "maxlag":            str(MAXLAG),
    "corr_duration":     str(CORR_DURATION),
    # Inter-station CC: classical ZZ
    "components_to_compute": "ZZ",
    "cc_type":           "CC",
    "whitening":         "A",   # "A" = apply to CC/SC, skip AC by design
    "winsorizing":       "3.0",
    # Single-station AC: PCC on Z autocorrelation
    "components_to_compute_single_station": "ZZ",
    "cc_type_single_station_AC": "PCC",
    "keep_all":  "Y",
    "keep_days": "Y",
    "stack_method": "linear",
}

for k, v in cc_params.items():
    update_config(db, k, v, category="cc", set_number=1)

print("cc_1 configured:")
print(f"  CC  inter-station  ZZ  (whitening=A: spectral whitening applied)")
print(f"  AC  single-station ZZ  (PCC — zero-lag spike suppressed)")
db.commit()

# %% [markdown]
# ## 6 · Configure Filter

# %%
# Filter: enable both CC and AC bands
for k, v in {
    "freqmin": str(FREQMIN),
    "freqmax": str(FREQMAX),
    "CC": "Y",
    "SC": "N",
    "AC": "Y",
}.items():
    update_config(db, k, v, category="filter", set_number=1)

db.commit()
print(f"filter_1: {FREQMIN}–{FREQMAX} Hz  (CC=Y, AC=Y)")

# %% [markdown]
# ## 7 · Configure the Three Refstack Sets

# %%
# Common to all three: full date range → use all available data
for sn in (ref_sn_linear, ref_sn_pws, ref_sn_tfpws):
    update_config(db, "ref_begin", STARTDATE, category="refstack", set_number=sn)
    update_config(db, "ref_end",   ENDDATE,   category="refstack", set_number=sn)
    update_config(db, "pws_power", str(PWS_POWER), category="refstack", set_number=sn)

# refstack_1 — linear mean
update_config(db, "stack_method", "linear", category="refstack", set_number=ref_sn_linear)
print(f"refstack_{ref_sn_linear}: linear  (arithmetic mean)")

# refstack_2 — phase-weighted stack (Schimmel & Paulssen 1997)
update_config(db, "stack_method",  "pws",        category="refstack", set_number=ref_sn_pws)
update_config(db, "pws_timegate",  str(PWS_TIMEGATE), category="refstack", set_number=ref_sn_pws)
print(f"refstack_{ref_sn_pws}: pws     (timegate={PWS_TIMEGATE}s, power={PWS_POWER})")

# refstack_3 — time-frequency phase-weighted stack (Schimmel & Gallart 2007)
update_config(db, "stack_method",  "tfpws",          category="refstack", set_number=ref_sn_tfpws)
update_config(db, "tfpws_nscales", str(TFPWS_NSCALES), category="refstack", set_number=ref_sn_tfpws)
print(f"refstack_{ref_sn_tfpws}: tfpws   (nscales={TFPWS_NSCALES}, power={PWS_POWER})")

db.commit()

# %% [markdown]
# ## 8 · Configure DataSource and Stations

# %%
DataSource = declare_tables().DataSource
ds = DataSource(
    name="local",
    uri=os.path.realpath(DATA_PATH),
    data_structure=DATA_STRUCTURE,
    auth_env="MSNOISE",
    network_code=NETWORK_CODE,
    channels=CHANNELS,
)
db.add(ds)
db.commit()

for net, sta, lon, lat, elev in STATIONS:
    update_station(db, net=net, sta=sta, X=lon, Y=lat, altitude=elev,
                   coordinates="DEG", used=1)
    print(f"  {net}.{sta}  ({lon:.3f}°E  {lat:.3f}°N)")

db.commit()

# %% [markdown]
# ## 9 · Build Workflow Graph
#
# Topology: `preprocess_1 → cc_1 → filter_1 → refstack_1/2/3`
# `stack` and `refstack` are **siblings** (both children of `filter`),
# so no `stack_1` is needed for this REF-only workflow.

# %%
created, existing, err = create_workflow_steps_from_config_sets(db)
assert err is None, f"Steps error: {err}"
print(f"Workflow steps: {created} created, {existing} existing")

created_links, existing_links, err = create_workflow_links_from_steps(db)
assert err is None, f"Links error: {err}"
print(f"Workflow links: {created_links} created, {existing_links} existing")

steps = {s.step_name: s for s in get_workflow_steps(db)}
print("Steps:", sorted(steps.keys()))

# %% [markdown]
# ## 10 · Scan Archive and Seed Jobs

# %%
scan_archive(init=True, threads=1)

# Update loc/chan on Station rows from DataAvailability
from sqlalchemy import text as _text
_db2 = connect()
for sta in _db2.query(declare_tables().Station):
    data = _db2.query(DataAvailability) \
        .filter(_text("net=:net")).filter(_text("sta=:sta")) \
        .group_by(DataAvailability.net, DataAvailability.sta,
                  DataAvailability.loc,  DataAvailability.chan) \
        .params(net=sta.net, sta=sta.sta).all()
    sta.used_location_codes = ",".join(sorted({d.loc  for d in data}))
    sta.used_channel_names  = ",".join(sorted({d.chan for d in data}))
_db2.commit(); _db2.close()

_db3 = connect()
n_da = _db3.query(DataAvailability).count()
print(f"DataAvailability rows: {n_da}")
_db3.close()

new_jobs(init=True)

# %% [markdown]
# ## 11 · Run the Pipeline
#
# No moving-stack step — `stack_refstack` reads directly from the
# daily CCFs written by `compute_cc` (when `keep_days = Y`).

# %%
print("── preprocess ─────────────────────────────────────────────────────────")
preprocess()

print("── new_jobs (after preprocess) ────────────────────────────────────────")
new_jobs(after="preprocess")

print("── compute_cc (CC inter-station + PCC autocorrelation) ────────────────")
compute_cc()

print("── new_jobs (after cc) ────────────────────────────────────────────────")
new_jobs(after="cc")

print("── stack_refstack: linear ─────────────────────────────────────────────")
stack_refstack()

print("Done.")

# %% [markdown]
# ## 12 · Gather MSNoiseResult Objects

# %%
db = connect()

# Base lineage through the daily stack (shared by all three refstacks)
base_ids = dict(preprocess=1, cc=1, filter=1)

result_linear = MSNoiseResult.from_ids(db, **base_ids, refstack=ref_sn_linear)
result_pws    = MSNoiseResult.from_ids(db, **base_ids, refstack=ref_sn_pws)
result_tfpws  = MSNoiseResult.from_ids(db, **base_ids, refstack=ref_sn_tfpws)

print("linear  lineage:", result_linear.lineage_names)
print("pws     lineage:", result_pws.lineage_names)
print("tfpws   lineage:", result_tfpws.lineage_names)

# %% [markdown]
# ## 13 · Retrieve Reference Stacks

# %%
ref_linear = result_linear.get_ref()
ref_pws    = result_pws.get_ref()
ref_tfpws  = result_tfpws.get_ref()

print(f"linear : {len(ref_linear)} REF(s) found")
print(f"pws    : {len(ref_pws)}    REF(s) found")
print(f"tfpws  : {len(ref_tfpws)}  REF(s) found")

common_keys = sorted(set(ref_linear) & set(ref_pws) & set(ref_tfpws))
print(f"\nCommon pairs across all three methods: {len(common_keys)}")
for key in common_keys:
    pair, comp = key
    print(f"  {pair}  {comp}")

# Separate CC pairs (two distinct stations) from AC pairs (same station)
cc_keys = [(p, c) for p, c in common_keys if p.split(":")[0] != p.split(":")[1]]
ac_keys = [(p, c) for p, c in common_keys if p.split(":")[0] == p.split(":")[1]]

print(f"\nCC pairs: {len(cc_keys)}   AC pairs: {len(ac_keys)}")

# %% [markdown]
# ## 14 · Helper: extract taxis and normalised waveform

# %%
def _get_waveform(ref_dict, key):
    """Return (taxis, waveform) from a get_ref() dict entry."""
    da = ref_dict[key]
    taxis = da.coords["taxis"].values if "taxis" in da.coords else np.arange(len(da))
    arr = da.values
    if arr.ndim > 1:
        arr = arr.squeeze()
    return taxis, arr


def _norm(arr):
    """Normalise to absolute maximum (returns unchanged if all-zero)."""
    m = np.abs(arr).max()
    return arr / m if m > 0 else arr


def _amplitude_spectrum(arr, fs):
    """One-sided amplitude spectrum up to Nyquist."""
    N = len(arr)
    A = np.abs(np.fft.rfft(arr))
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    return f, A / A.max() if A.max() > 0 else A

# %% [markdown]
# ## 15 · Figure 1 — Time Domain Comparison
#
# Each row is one pair (CC or AC).  The three REF stacks are built directly
# from the daily CCFs (`keep_days=Y`) without an intermediate moving stack.
# Waveforms are normalised to their absolute maximum.

# %%
COLORS  = {"linear": "#1f77b4", "pws": "#ff7f0e", "tfpws": "#2ca02c"}
LABELS  = {"linear": "Linear",  "pws": "PWS",     "tfpws": "tf-PWS"}

all_keys   = cc_keys + ac_keys
n_rows     = max(1, len(all_keys))
fig1, axes1 = plt.subplots(n_rows, 1, figsize=(12, 3.2 * n_rows), squeeze=False)

fig1.suptitle(
    f"Reference Stack Comparison — Time Domain\n"
    f"Filter {FREQMIN}–{FREQMAX} Hz  |  {STARTDATE}",
    fontsize=13, y=1.01,
)

for row, key in enumerate(all_keys):
    pair, comp = key
    ax = axes1[row, 0]

    for method, ref_dict in [
        ("linear", ref_linear),
        ("pws",    ref_pws),
        ("tfpws",  ref_tfpws),
    ]:
        if key not in ref_dict:
            continue
        taxis, arr = _get_waveform(ref_dict, key)
        ax.plot(taxis, _norm(arr),
                color=COLORS[method], lw=1.1, alpha=0.85,
                label=LABELS[method])

    kind = "AC" if pair.split(":")[0] == pair.split(":")[1] else "CC"
    ax.set_title(f"{pair}  [{comp}]  — {kind}", fontsize=11)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Norm. amplitude")
    ax.set_xlim(-MAXLAG, MAXLAG)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 16 · Figure 2 — Frequency Domain Comparison
#
# Amplitude spectra of the three REF stacks for each pair.  The passband
# of `filter_1` is shaded in grey.  Differences reflect the coherence
# weighting applied by PWS and tf-PWS within the band.

# %%
FS = CC_SAMPLING_RATE   # sampling rate of the stacked CCFs

fig2, axes2 = plt.subplots(n_rows, 1, figsize=(12, 3.2 * n_rows), squeeze=False)

fig2.suptitle(
    f"Reference Stack Comparison — Frequency Domain\n"
    f"Filter {FREQMIN}–{FREQMAX} Hz  |  {STARTDATE}",
    fontsize=13, y=1.01,
)

F_PLOT_MAX = FREQMAX * 2.0   # show up to 2× freqmax

for row, key in enumerate(all_keys):
    pair, comp = key
    ax = axes2[row, 0]

    for method, ref_dict in [
        ("linear", ref_linear),
        ("pws",    ref_pws),
        ("tfpws",  ref_tfpws),
    ]:
        if key not in ref_dict:
            continue
        _, arr = _get_waveform(ref_dict, key)
        f, A = _amplitude_spectrum(arr, FS)
        ax.plot(f, A, color=COLORS[method], lw=1.1, alpha=0.85,
                label=LABELS[method])

    # Shade the filter passband
    ax.axvspan(FREQMIN, FREQMAX, alpha=0.08, color="grey", label="Passband")

    kind = "AC" if pair.split(":")[0] == pair.split(":")[1] else "CC"
    ax.set_title(f"{pair}  [{comp}]  — {kind}", fontsize=11)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Norm. amplitude")
    ax.set_xlim(0, F_PLOT_MAX)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 17 · Figure 3 — Side-by-side CC vs AC (one method per column)
#
# A compact overview: rows = stack method, columns = CC / AC.
# Waveforms are normalised; the zero-lag line is marked.

# %%
if cc_keys and ac_keys:
    n_methods = 3
    methods   = [("linear", ref_linear), ("pws", ref_pws), ("tfpws", ref_tfpws)]

    # Pick the first CC pair and the first AC pair for the overview
    cc_key0 = cc_keys[0]
    ac_key0 = ac_keys[0]

    fig3, axes3 = plt.subplots(
        n_methods, 2,
        figsize=(13, 3.0 * n_methods),
        squeeze=False,
        sharex="col",
    )
    fig3.suptitle(
        "REF stack by method: CC (left) vs AC (right)\n"
        f"Filter {FREQMIN}–{FREQMAX} Hz  |  {STARTDATE}",
        fontsize=13, y=1.01,
    )

    for row, (method, ref_dict) in enumerate(methods):

        # ── CC ────────────────────────────────────────────────────────────
        ax = axes3[row, 0]
        if cc_key0 in ref_dict:
            taxis, arr = _get_waveform(ref_dict, cc_key0)
            ax.plot(taxis, _norm(arr), color=COLORS[method], lw=1.0)
        ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.4)
        ax.set_ylabel(LABELS[method], fontsize=11)
        ax.set_xlim(-MAXLAG, MAXLAG)
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.set_title(f"CC  |  {cc_key0[0]}  [{cc_key0[1]}]", fontsize=10)
        if row == n_methods - 1:
            ax.set_xlabel("Lag (s)")

        # ── AC ────────────────────────────────────────────────────────────
        ax = axes3[row, 1]
        if ac_key0 in ref_dict:
            taxis, arr = _get_waveform(ref_dict, ac_key0)
            ax.plot(taxis, _norm(arr), color=COLORS[method], lw=1.0)
        ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.4)
        ax.set_xlim(0, MAXLAG)   # AC is symmetric; show positive lags only
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.set_title(f"AC  |  {ac_key0[0]}  [{ac_key0[1]}]", fontsize=10)
        if row == n_methods - 1:
            ax.set_xlabel("Lag (s)")

    plt.tight_layout()
    plt.show()

else:
    print("Skipping Figure 3: need at least one CC pair and one AC pair.")

# %% [markdown]
# ## Notes
#
# ### Interpreting the differences between stack methods
#
# * **Linear** — arithmetic mean of all window CCFs.  Fast and transparent.
#   In the frequency domain the spectrum reflects the raw energy distribution
#   of the ambient noise field within the passband.
#
# * **PWS** (`pws`) — time-domain phase coherence weights downweight
#   windows whose instantaneous phase is inconsistent with the ensemble.
#   Transient signals (earthquakes, cultural noise) contribute random phases
#   and are attenuated.  The result is often cleaner in the time domain but
#   may suppress weak coherent arrivals if `pws_power` is too large.
#
# * **tf-PWS** (`tfpws`) — same idea but the coherence is evaluated at each
#   CWT scale independently.  This gives frequency-selective suppression:
#   narrow-band transients are attenuated only in the frequency bands where
#   they are incoherent.  For AC studies (Romero & Schimmel 2018) the
#   advantage of tf-PWS over PWS is most visible when the target reflection
#   is narrow-band (e.g. a basement reflection at 3–12 Hz embedded in
#   broadband cultural noise).
#
# ### Why AC uses PCC instead of CC
#
# `cc_type_single_station_AC = "PCC"` engages the PCC2 algorithm for the
# autocorrelation.  PCC operates on instantaneous phase only, which
# eliminates the dominant zero-lag spike inherent to the classical
# autocorrelogram without requiring explicit muting or spectral division.
# This is essential for detecting shallow reflections at lags < 1 s.
#
# ### Whitening behaviour with `whitening = "A"`
#
# `whitening = "A"` (auto) means spectral whitening is applied to
# inter-station CC and SC but **skipped for AC by design**: the AC branch
# in `s03_compute_no_rotation` unconditionally starts from `_data_bp`
# (the bandpassed, non-whitened copy of the traces).  There is therefore no
# risk of collapsing the ACF to a sinc function regardless of the whitening
# setting — `"A"` is the correct and recommended value when computing both
# CC and AC in the same step.