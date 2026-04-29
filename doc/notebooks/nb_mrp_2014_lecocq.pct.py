# %% [markdown]
# # MRP — 2014 Lecocq *et al.* (MSNoise/UnderVolc)
#
# This notebook reproduces figures from:
#
# > Lecocq, T., Caudron, C., & Brenguier, F. (2014). MSNoise, a Python Package
# > for Monitoring Seismic Velocity Changes Using Ambient Seismic Noise.
# > *Seismological Research Letters*, 85(3), 715–726.
# > https://doi.org/10.1785/0220130073
#
# It uses the **MSNoise Reproducible Papers** (MRP) registry to download
# pre-computed results at the `stack` and `dvv` levels, so no data download
# or pipeline run is needed.
#
# **Sections**
# 1. Load the paper via MRP
# 2. Reference stack — distance plot
# 3. Moving stack — interferogram (one pair)
# 4. DVV — all mov_stack results from `mwcs_dtt_dvv`

# %% [markdown]
# ## 0. Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
from pathlib import Path

from msnoise.papers import MRP
from msnoise.core.stations import distance_from_coords

# %% [markdown]
# ## 1. Load paper from MRP registry

# %%
mrp = MRP()
p = mrp.get_paper("2014_Lecocq_MSNoiseUndervolc")
p.info()

# %%
# Download stack level (refstack + stack CCFs, ~7.8 GB cached after first run)
stack = p.get_project("stack")

# List available results
ref_results = stack.list("refstack")
stack_results = stack.list("stack")
print("refstack results:", ref_results)
print("stack results   :", stack_results)

# %%
s_ref   = ref_results[0]
s_stack = stack_results[0]

# %% [markdown]
# ## 2. Reference stack — distance plot
#
# All ZZ inter-station reference CCFs sorted by interstation distance.

# %%
# Helper: load station coordinates from project.yaml
def _station_coords(result):
    """Walk up from output_folder until project.yaml is found."""
    search = Path(result.output_folder)
    for candidate in [search, *search.parents]:
        p = candidate / "project.yaml"
        if p.exists():
            with open(p) as fh:
                doc = yaml.safe_load(fh)
            return {f"{s['net']}.{s['sta']}": s for s in doc.get("stations", [])}
    raise FileNotFoundError("project.yaml not found")


def pair_dist(pair, coords):
    a, b = pair.split(":")
    ka = ".".join(a.split(".")[:2])
    kb = ".".join(b.split(".")[:2])
    sa, sb = coords[ka], coords[kb]
    return distance_from_coords(
        sa["X"], sa["Y"], sb["X"], sb["Y"], sa.get("coordinates", "DEG")
    )


def plot_distance(result, r, comp="ZZ", scale=0.8, maxlag=30,
                  color_pos="steelblue", color_neg="tomato", ax=None):
    """Wiggle distance plot from get_ref() or get_ccf() output dict."""
    coords = _station_coords(result)
    taxis  = list(r.values())[0].coords["taxis"].values
    mask   = np.abs(taxis) <= maxlag
    t      = taxis[mask]

    rows = []
    for (pair, c), da in r.items():
        if c != comp:
            continue
        try:
            d = pair_dist(pair, coords)
        except KeyError:
            continue
        rows.append((d, pair, da.values[mask]))
    rows.sort()

    if not rows:
        raise ValueError(f"No pairs found for comp={comp!r}")

    dists = [d for d, _, _ in rows]
    dy    = (max(dists) - min(dists)) / max(len(rows) - 1, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.figure

    for dist, pair, tr in rows:
        norm = np.nanmax(np.abs(tr)) or 1.0
        w    = tr / norm * dy * scale
        ax.plot(t, w + dist, "k", lw=0.5, alpha=0.8)
        ax.fill_between(t, dist, w + dist, where=w > 0,
                        color=color_pos, alpha=0.6, interpolate=True)
        ax.fill_between(t, dist, w + dist, where=w < 0,
                        color=color_neg, alpha=0.4, interpolate=True)

    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.3)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Distance (km)")
    ax.set_xlim(t[0], t[-1])
    plt.tight_layout()
    return fig, ax

# %%
r_ref = s_ref.get_ref()
fig, ax = plot_distance(s_ref, r_ref, comp="ZZ", maxlag=30, scale=0.8)
ax.set_title("Reference stacks — ZZ (2014 Lecocq / UnderVolc)")
plt.show()

# %% [markdown]
# ## 3. Moving stack — interferogram
#
# Daily 30-day rolling CCF for one pair, plotted as an image (time × lag).

# %%
# Pick the pair with the best SNR (longest baseline visible in distance plot)
PAIR     = "YA.UV05.00:YA.UV10.00"
COMP     = "ZZ"
MOV      = ("30D", "1D")
MAXLAG   = 15.0

r_ccf = s_stack.get_ccf(PAIR, COMP, MOV)

taxis = r_ccf.coords["taxis"].values
times = r_ccf.coords["times"].values
mask  = np.abs(taxis) <= MAXLAG
t     = taxis[mask]

mat = r_ccf.values[:, mask]          # (n_days, n_lag)
# Normalise each row for display
norms = np.nanmax(np.abs(mat), axis=1, keepdims=True)
norms[norms == 0] = 1.0
mat_n = mat / norms

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(
    mat_n.T,
    aspect="auto",
    origin="lower",
    extent=[mdates.date2num(times[0].astype("M8[ms]").astype(object)),
            mdates.date2num(times[-1].astype("M8[ms]").astype(object)),
            t[0], t[-1]],
    cmap="seismic",
    vmin=-1, vmax=1,
    interpolation="nearest",
)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
ax.set_ylabel("Lag (s)")
ax.set_title(f"Moving stack {MOV} — {PAIR} {COMP}")
plt.colorbar(im, ax=ax, label="Norm. amplitude")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. DVV — all `mwcs_dtt_dvv` mov_stack results
#
# Download the `dvv` level (small, ~300 MB) and plot aggregate dv/v for
# every available moving stack on the same axes.

# %%
dvv_proj = p.get_project("dvv")
dvv_results = dvv_proj.list("mwcs_dtt_dvv")
print("mwcs_dtt_dvv results:", dvv_results)

# %%
d = dvv_results[0]
dvv_all = d.get_dvv(pair_type="CC", components="ZZ")   # dict {(pair_type, comp, mov_stack): Dataset}
print("Available keys:", list(dvv_all.keys()))

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
ax_dvv, ax_coh = axes

CMAP = plt.cm.viridis
keys = sorted(dvv_all.keys(), key=lambda k: k[1][0])   # sort by mov_stack duration
colors = CMAP(np.linspace(0.15, 0.85, len(keys)))

for (pt, comp, ms), ds, color in zip(keys, [dvv_all[k] for k in keys], colors):
    label = f"{ms[0]}/{ms[1]}"
    times = ds["times"].values.astype("M8[ms]").astype(object)

    dvv  = ds["dvv"].values * 100      # → %
    err  = ds["err"].values * 100
    coh  = ds.get("coh", ds.get("m", None))

    ax_dvv.plot(times, dvv, color=color, lw=1.2, label=label)
    ax_dvv.fill_between(times, dvv - err, dvv + err,
                        color=color, alpha=0.15)

    if coh is not None:
        ax_coh.plot(times, coh.values, color=color, lw=1.0)

ax_dvv.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
ax_dvv.set_ylabel("dv/v (%)")
ax_dvv.legend(title="mov_stack", ncol=3, fontsize=8)
ax_dvv.set_title("dv/v — mwcs_dtt_dvv ZZ ALL pairs (2014 Lecocq / UnderVolc)")

ax_coh.set_ylabel("Coherence / m")
ax_coh.set_xlabel("Date")
ax_coh.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
