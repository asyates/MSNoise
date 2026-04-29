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
# MSNoise Reproducible Papers — Analysis Notebook

This notebook demonstrates the three entry paths into `MSNoiseProject` and
then shows canonical analyses that work identically regardless of how the
project was loaded:

| Section | What it does |
|---------|-------------|
| **0 — Load** | Path A (live), B (local archive), C (MRP registry) |
| **1 — CCF matrix** | `project.list("stack")` → waveform overview per pair |
| **2 — REF stacks** | `result.branches()` → refstack waveform comparison |
| **3 — dv/v timeseries** | `project.list("dvv")` → network-level velocity changes |

## Prerequisites

* MSNoise ≥ 2.0 installed (including `pooch`, `zstandard`, `platformdirs`).
* For **Path A**: a live project directory with `db.ini`.
* For **Path B**: a `.tar.zst` project archive (produced by
  `msnoise project export`).
* For **Path C**: network access to the MRP registry; the paper must have a
  `bundle_pointer.yaml` with the requested level.
"""

# %% [markdown]
# ## 0 · Load the project
#
# Run exactly **one** of the three cells below, then continue from Section 1.

# %% [markdown]
# ### Path A — live project (cwd has `db.ini`)

# %%
# from msnoise.project import MSNoiseProject
# project = MSNoiseProject.from_current(".")   # or pass an explicit path
# print(project.project_dir)

# %% [markdown]
# ### Path B — local project archive

# %%
# from msnoise.project import MSNoiseProject
# project = MSNoiseProject.from_archive("level_stack.tar.zst")
# print(project.project_dir)

# %% [markdown]
# ### Path C — MSNoise Reproducible Papers registry  *(default for this notebook)*

# %%
from msnoise.papers import MRP

mrp = MRP()
mrp.list_papers()

# %%
# +
PAPER_ID = "2016_DePlaen_PitonDeLaFournaise"   # ← change as needed
LEVEL    = "stack"                               # "stack" | "dvv" | …

paper = mrp.get_paper(PAPER_ID)
paper.info()
# -

# %%
project = paper.get_project(LEVEL)
print("Project dir:", project.project_dir)

# %% [markdown]
# ## 1 · CCF matrix overview
#
# Load all stacked CCF results for the requested filter / mov_stack combination
# and plot an overview matrix: one row per pair, lag time on the x-axis.

# %%
# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Parameters — adjust to match the paper's filter / mov_stack settings
COMPONENT = "ZZ"
MOV_STACK = ("1D", "1D")    # (duration, step) as stored in the NetCDF

results = project.list("stack")
print(f"Found {len(results)} stack lineage(s).")

fig, axes = plt.subplots(
    len(results), 1,
    figsize=(12, 2.5 * max(len(results), 1)),
    squeeze=False,
)

for ax, result in zip(axes[:, 0], results):
    try:
        ds = result.get_ccf(component=COMPONENT, mov_stack=MOV_STACK)
    except Exception as exc:
        ax.set_visible(False)
        print(f"  Skipped {result.lineage_names[-1]}: {exc}")
        continue

    pairs = ds.coords["pair"].values if "pair" in ds.coords else ds.coords.get("pairs", [None]).values
    times = ds.coords.get("times", None)
    taxis = ds.coords.get("taxis", None)

    if taxis is None or times is None:
        ax.set_visible(False)
        continue

    data = ds["ccf"].values if "ccf" in ds else ds[list(ds.data_vars)[0]].values
    # data shape: (times, pairs, taxis) or (pairs, taxis) for REF
    if data.ndim == 3:
        data = data.mean(axis=0)   # collapse times → (pairs, taxis)

    t = taxis.values
    for i, (row, pair) in enumerate(zip(data, pairs)):
        norm = np.abs(row).max() or 1.0
        ax.plot(t, row / norm + i, lw=0.8, color="steelblue")
        ax.text(t[-1] + 0.1, i, str(pair), fontsize=6, va="center")

    ax.set_title(f"{result.lineage_names[-1]}  ({COMPONENT}, {MOV_STACK})", fontsize=9)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Pair index")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    if hasattr(ds, "close"):
        ds.close()

fig.tight_layout()
plt.show()
# -

# %% [markdown]
# ## 2 · REF stack comparison
#
# For each stack lineage, traverse to the sibling refstack(s) and overlay
# their reference waveforms for a selected pair.

# %%
# +
PAIR = None    # set to e.g. "BE.UCC..HHZ:BE.MEM..HHZ" or leave None for first pair

fig2, ax2 = plt.subplots(figsize=(12, 4))

for result in results:
    try:
        branches = result.branches()
    except Exception:
        branches = []

    for branch in branches:
        if branch.category != "refstack":
            continue
        try:
            ds_ref = branch.get_ref(component=COMPONENT)
        except Exception:
            continue

        pairs_in = ds_ref.coords.get("pair", ds_ref.coords.get("pairs", None))
        if pairs_in is None:
            continue

        target_pair = PAIR or str(pairs_in.values[0])
        taxis_ref = ds_ref.coords.get("taxis", None)
        if taxis_ref is None:
            continue

        idx = list(pairs_in.values).index(target_pair) if target_pair in pairs_in.values else 0
        waveform = ds_ref["ref"].values[idx] if "ref" in ds_ref else ds_ref[list(ds_ref.data_vars)[0]].values[idx]
        norm = np.abs(waveform).max() or 1.0
        ax2.plot(taxis_ref.values, waveform / norm,
                 label=f"{branch.lineage_names[-1]}", alpha=0.8, lw=1.2)

        if hasattr(ds_ref, "close"):
            ds_ref.close()

ax2.set_title(f"REF stack comparison — pair: {PAIR or '(first available)'} — {COMPONENT}", fontsize=10)
ax2.set_xlabel("Lag (s)")
ax2.set_ylabel("Normalised amplitude")
ax2.legend(fontsize=8, loc="upper right")
ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator())
fig2.tight_layout()
plt.show()
# -

# %% [markdown]
# ## 3 · dv/v timeseries
#
# Load the final DVV aggregates (requires a project loaded at level `"dvv"`).
# If the current project was loaded at the `"stack"` level the cell below
# gracefully skips with a message.

# %%
# +
dvv_results = project.list("mwcs_dtt_dvv")

if not dvv_results:
    print(
        "No mwcs_dtt_dvv outputs found in this project.\n"
        "Load at level='dvv' to access the final dv/v aggregates:\n"
        "    project = paper.get_project('dvv')"
    )
else:
    fig3, ax3 = plt.subplots(figsize=(12, 4))

    for dvv_result in dvv_results:
        try:
            ds_dvv = dvv_result.get_dvv(component=COMPONENT, mov_stack=MOV_STACK)
        except Exception as exc:
            print(f"  Skipped {dvv_result.lineage_names[-1]}: {exc}")
            continue

        if "times" not in ds_dvv.coords or "dvv" not in ds_dvv:
            ds_dvv.close()
            continue

        t = ds_dvv.coords["times"].values.astype("datetime64[D]")
        dvv_mean = float(ds_dvv["dvv"].mean("pair")) if "pair" in ds_dvv.dims else ds_dvv["dvv"].values
        err_mean = float(ds_dvv["err"].mean("pair")) if ("err" in ds_dvv and "pair" in ds_dvv.dims) else None

        ax3.plot(t, dvv_mean * 100, label=dvv_result.lineage_names[-1], lw=1.2)
        if err_mean is not None:
            ax3.fill_between(t,
                             (dvv_mean - err_mean) * 100,
                             (dvv_mean + err_mean) * 100,
                             alpha=0.2)
        ds_dvv.close()

    ax3.axhline(0, color="k", lw=0.6, ls="--")
    ax3.set_title(f"dv/v — {COMPONENT}  {MOV_STACK}", fontsize=10)
    ax3.set_xlabel("Date")
    ax3.set_ylabel("dv/v (%)")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    plt.show()
# -
