#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_plot_map.py — one figure per category showing how the fiscal profile of
every hromada moves between clusters over 2018-2025, plus a summary panel
counting how often each community changed.

It does NOT re-run the pipeline. Everything it needs is already on disk:

    article_outputs/extra/clusters_<cat>_all_years.csv   <- written by run_all.py
    article_outputs/cache_<cat>.pkl                      <- for the budget totals
    viz_dashboard_data2/l3.geojson                       <- hromada borders
    ukr_admin0.geojson                                   <- backdrop (optional)

COLLAPSED BUDGETS. A community whose budget has collapsed — schools evacuated,
services gone, transfers stopped — keeps only a residue of local taxes. Because
the cluster score gives transfers a negative weight, such a budget scores as the
most self-sufficient one in the country, which is the opposite of the truth.
These community-years are detected by their total (below COLLAPSED_BELOW of the
year's median) and drawn in a separate grey instead of a cluster colour.

Usage:
    python3 make_plot_map.py                  # revenues
    python3 make_plot_map.py --cat prog       # programme expenditures
    python3 make_plot_map.py --cat all        # every category that has a file
"""

import os
import re
import ast
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS — the only block you normally edit
# =============================================================================

OUT_DIR    = "article_outputs"
GEOJSON_L3 = "viz_dashboard_data2/l3.geojson"
GEOJSON_UA = "ukr_admin0.geojson"        # backdrop; set to None to skip
GEO_KEY    = "BUDGET_CODE"               # property in l3.geojson holding the code
NCOLS      = 3                           # panels per row

# a community-year whose total is below this fraction of the year's median is
# treated as a collapsed / non-reporting budget, not as a fiscal profile
COLLAPSED_BELOW = 0.10

COL_NODATA    = "#ececec"    # community does not exist yet / no record at all
COL_COLLAPSED = "#9a9a9a"    # record exists but the budget has collapsed

LABELS = {"income": "revenue", "prog": "programme expenditure",
          "econ": "expenditure (economic classification)",
          "func": "expenditure (functional classification)"}
# only these are renumbered by financial score in run_all.py, so only for these
# does a green-to-red ramp mean anything; the others get a categorical palette
ORDERED = {"income", "prog"}

RUN_ALL = "run_all.py"       # settings are read from here, never duplicated

ap = argparse.ArgumentParser()
ap.add_argument("--cat", default="all",
                choices=["income", "prog", "econ", "func", "all"],
                help="which clustering to map (default: all)")
ap.add_argument("--out", default=None, help="output png path (single category only)")
ARGS = ap.parse_args()

# =============================================================================

import geopandas as gpd          # imported here so the error message is clear
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def pipeline_settings(path=RUN_ALL):
    """Read the few constants that define the clustering straight out of
    run_all.py, so this script can never drift away from the pipeline. The file
    is parsed, not executed, so nothing is re-run."""
    want = {"N_PC": 10, "K": 10, "SEED": 42, "CENTER_BY_YEAR": True,
            "MIN_MEAN_SHARE": 0.0, "MIN_PRESENCE": 0.0}
    if not os.path.exists(path):
        print(f"   WARNING {path} not found; using defaults {want}")
        return want
    txt = open(path, encoding="utf-8").read()
    for name in list(want):
        m = re.search(rf"^{name}\s*=\s*([^#\n]+)", txt, re.M)
        if m:
            try:
                want[name] = ast.literal_eval(m.group(1).strip())
            except Exception:
                pass
    print(f"   settings read from {path}: " +
          "  ".join(f"{k}={v}" for k, v in want.items()))
    return want


def cluster_from_cache(cat, cfg):
    """Build the clusters for a category that run_all.py did not cluster.
    Same steps and same settings as the pipeline: shares -> trim -> centre ->
    standardise -> PCA -> k-means. Cluster numbers are arbitrary here, because
    the financial score that orders them is defined only for income and prog."""
    cache = os.path.join(OUT_DIR, f"cache_{cat}.pkl")
    if not os.path.exists(cache):
        print(f"   cannot build '{cat}': {cache} not found")
        return None
    print(f"   building clusters for '{cat}' from {cache}")
    obj = pd.read_pickle(cache)
    df = obj[0] if isinstance(obj, tuple) else obj

    pv = df.pivot_table(index=["BUDGET_CODE", "Year"], columns="Code",
                        values="Amount", aggfunc="sum", fill_value=0)
    shares = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    n_all = shares.shape[1]

    keep = ((shares.mean() >= cfg["MIN_MEAN_SHARE"]) &
            ((shares > 0).mean() >= cfg["MIN_PRESENCE"]))
    if keep.any():
        money = 100 * shares.loc[:, keep].sum().sum() / shares.sum().sum()
        shares = shares.loc[:, keep]
        print(f"   items: {shares.shape[1]} kept of {n_all}, "
              f"{money:.2f}% of the money covered")
    if cfg["CENTER_BY_YEAR"]:
        yrs = shares.index.get_level_values("Year")
        shares = shares - shares.groupby(yrs).transform("mean")
        print("   centred within each year")

    X = StandardScaler().fit_transform(shares)
    n_pc = min(cfg["N_PC"], shares.shape[1], shares.shape[0])
    Z = PCA(n_components=n_pc, random_state=cfg["SEED"]).fit_transform(X)
    lab = KMeans(n_clusters=cfg["K"], random_state=cfg["SEED"],
                 n_init=10).fit_predict(Z) + 1

    out = shares.index.to_frame(index=False)
    out["Cluster"] = lab
    dst = os.path.join(OUT_DIR, "extra", f"clusters_{cat}_all_years.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"   wrote {dst}  ({len(out)} community-years, k={cfg['K']}, "
          f"{n_pc} components)")
    return dst



def norm_key(s):
    """Budget codes are 10-digit strings; Excel and geojson both like to drop
    leading zeros or turn them into floats. Normalise both sides the same way."""
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(10)


def load_geo():
    print(f"reading {GEOJSON_L3}")
    g = gpd.read_file(GEOJSON_L3)
    if GEO_KEY not in g.columns:
        cand = [c for c in g.columns if "BUDGET" in c.upper() or "COD" in c.upper()]
        raise SystemExit(f"'{GEO_KEY}' not in the geojson. Candidates: {cand}")
    g[GEO_KEY] = norm_key(g[GEO_KEY])
    g = g[g.geometry.notna()].copy()
    print(f"   {len(g)} polygons")
    b = gpd.read_file(GEOJSON_UA) if (GEOJSON_UA and os.path.exists(GEOJSON_UA)) else None
    return g, b


def collapsed_years(cat):
    """(BUDGET_CODE, Year) pairs whose total is a residue rather than a budget."""
    cache = os.path.join(OUT_DIR, f"cache_{cat}.pkl")
    if not os.path.exists(cache):
        print(f"   no {cache}; collapsed budgets not detected for '{cat}'")
        return set(), None
    obj = pd.read_pickle(cache)
    df = obj[0] if isinstance(obj, tuple) else obj
    tot = df.groupby(["BUDGET_CODE", "Year"])["Amount"].sum().reset_index()
    tot["BUDGET_CODE"] = norm_key(tot["BUDGET_CODE"])
    med = tot.groupby("Year")["Amount"].transform("median")
    tot["Collapsed"] = tot["Amount"] < COLLAPSED_BELOW * med
    flagged = tot[tot["Collapsed"]]
    print(f"   collapsed budgets (total below {COLLAPSED_BELOW:.0%} of the year's "
          f"median): {len(flagged)} community-years")
    if len(flagged):
        by_year = flagged.groupby("Year").size().to_dict()
        print(f"   by year: {by_year}")
    return set(zip(flagged["BUDGET_CODE"], flagged["Year"])), tot


def make_map(cat, gdf, back):
    src = os.path.join(OUT_DIR, "extra", f"clusters_{cat}_all_years.csv")
    print(f"\n=== {cat} ===")
    if not os.path.exists(src):
        print(f"   {src} not found — run_all.py did not cluster this category")
        if cluster_from_cache(cat, CFG) is None:
            return
    png = ARGS.out or os.path.join(OUT_DIR, f"fig6_clusters_map_{cat}.png")
    label = LABELS.get(cat, cat)

    print(f"reading {src}")
    cl = pd.read_csv(src)
    cl["BUDGET_CODE"] = norm_key(cl["BUDGET_CODE"])
    cl["Year"] = cl["Year"].astype(int)
    years = sorted(cl["Year"].unique())
    kmax = int(cl["Cluster"].max())
    print(f"   {cl['BUDGET_CODE'].nunique()} communities, {years[0]}-{years[-1]}, "
          f"{kmax} clusters")

    bad, _ = collapsed_years(cat)
    cl["Collapsed"] = [(b, y) in bad for b, y in zip(cl["BUDGET_CODE"], cl["Year"])]

    # changes are counted over the years in which the budget actually worked, so
    # a community that collapses and comes back does not score two "changes"
    ok = cl[~cl["Collapsed"]].sort_values(["BUDGET_CODE", "Year"])
    ok["Changed"] = (ok.groupby("BUDGET_CODE")["Cluster"].diff().fillna(0) != 0).astype(int)
    changes = ok.groupby("BUDGET_CODE")["Changed"].sum().rename("Changes")
    print(f"   cluster changes per community: median {changes.median():.0f}, "
          f"max {changes.max():.0f}")

    matched = gdf[GEO_KEY].isin(set(cl["BUDGET_CODE"])).sum()
    print(f"   matched to the clustering: {matched} of {len(gdf)} polygons "
          f"({100 * matched / len(gdf):.1f}%)")
    if matched == 0:
        raise SystemExit("No polygon matched. Check GEO_KEY and the code format.")

    if cat in ORDERED:
        cmap = ListedColormap(plt.cm.RdYlGn_r(np.linspace(0.08, 0.95, kmax)))
        note = (" — most self-sufficient", " — most transfer-dependent")
    else:                        # arbitrary k-means numbering: no ramp, no order
        cmap = ListedColormap(plt.cm.tab10(np.arange(kmax) % 10))
        note = ("", "")
    norm = BoundaryNorm(np.arange(0.5, kmax + 1.5), kmax)

    panels = list(years) + ["changes"]
    nrows = int(np.ceil(len(panels) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(5.6 * NCOLS, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, panel in zip(axes, panels):
        if back is not None:
            back.plot(ax=ax, color="#f6f6f6", edgecolor="#cccccc", linewidth=0.4)
        gdf.plot(ax=ax, color=COL_NODATA, edgecolor="white", linewidth=0.08)

        if panel == "changes":
            m = gdf.merge(changes, left_on=GEO_KEY, right_index=True, how="inner")
            m.plot(column="Changes", cmap="YlOrBr", vmin=0, vmax=float(changes.max()),
                   linewidth=0.08, edgecolor="white", ax=ax, legend=True,
                   legend_kwds={"shrink": 0.55,
                                "label": f"cluster changes, {years[0]}–{years[-1]}"})
            ax.set_title(f"How often the {label} profile changed\n"
                         f"{years[0]}–{years[-1]} (collapsed years excluded)",
                         fontsize=10.5, fontweight="bold")
        else:
            sub = cl[cl["Year"] == panel]
            good = gdf.merge(sub[~sub["Collapsed"]][["BUDGET_CODE", "Cluster"]],
                             left_on=GEO_KEY, right_on="BUDGET_CODE", how="inner")
            dead = gdf.merge(sub[sub["Collapsed"]][["BUDGET_CODE"]],
                             left_on=GEO_KEY, right_on="BUDGET_CODE", how="inner")
            if len(good):
                good.plot(column="Cluster", cmap=cmap, norm=norm, linewidth=0.08,
                          edgecolor="white", ax=ax)
            if len(dead):
                dead.plot(ax=ax, color=COL_COLLAPSED, edgecolor="white", linewidth=0.08)
            ax.set_title(f"{panel}  ({len(good)} reporting, {len(dead)} collapsed)",
                         fontsize=11)
        ax.set_axis_off()

    for ax in axes[len(panels):]:
        ax.set_axis_off()

    handles = [Patch(facecolor=cmap(i), edgecolor="white",
                     label=f"Cluster {i + 1}" + (note[0] if i == 0 else
                                                 note[1] if i == kmax - 1 else ""))
               for i in range(kmax)]
    handles += [
        Patch(facecolor=COL_COLLAPSED, edgecolor="white",
              label=f"collapsed budget (total below {COLLAPSED_BELOW:.0%} of the "
                    f"year's median)"),
        Patch(facecolor=COL_NODATA, edgecolor="#cccccc",
              label="no data (not yet amalgamated, occupied or not reporting)")]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, kmax // 2 + 1),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    extra = "" if cat in ORDERED else "  (cluster numbers are arbitrary for this category)"
    fig.suptitle(f"Fiscal profiles of Ukrainian hromadas, {label} clusters "
                 f"{years[0]}–{years[-1]}{extra}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0.045, 1, 0.985))
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}  ({os.path.getsize(png) / 1024:.0f} kB)")

    tab = cl[~cl["Collapsed"]].groupby(["Year", "Cluster"]).size().unstack(fill_value=0)
    tab["collapsed"] = cl[cl["Collapsed"]].groupby("Year").size()
    tab = tab.fillna(0).astype(int)
    tab.to_csv(os.path.join(OUT_DIR, "extra", f"cluster_counts_by_year_{cat}.csv"))
    print(tab.to_string())


CFG = pipeline_settings()
gdf, back = load_geo()
cats = ["income", "prog", "econ", "func"] if ARGS.cat == "all" else [ARGS.cat]
if ARGS.cat == "all" and ARGS.out:
    raise SystemExit("--out works for a single category only")
for c in cats:
    make_map(c, gdf, back)
print("\ndone")