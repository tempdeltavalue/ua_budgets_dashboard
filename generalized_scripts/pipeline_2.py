#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — one script that regenerates EVERY table and figure of the manuscript
"Data-driven pipeline for local budget profiling under systemic shocks in Ukraine".

One run -> out_dir/ with exactly one file per number in the manuscript:

    report.html          <- open this: all figures and tables on one page
    manifest.csv         <- manuscript number -> file name
    summary.txt          <- full text log (numbers, diagnostics, warnings)

    Table_1 .. Table_5, Table_A_1 .. Table_A_3   (.csv + .xlsx)
    fig1 .. fig10, figA1 (+ optional figB1/figB2 Poland)   (.png)
    extra/               <- working data and the supplementary tables A.4 - A.9

A figure that exists for both budget sides (revenues and program expenditures) is
ONE file with one panel per side — never fig..._income.png plus fig..._prog.png.
Every .png is rendered to memory, written once and its size read back, so an empty
file is reported in the OUTPUT AUDIT at the end of summary.txt instead of being
discovered later in a file manager.

Everything comes from ONE clustering run per category, so cluster numbers are
identical in every table and figure. Fix SEED and the numbers are reproducible.

Usage:
    python run_all.py                 # uses the cache if it exists
    python run_all.py --reload        # re-reads all parquet files
    python run_all.py --quick         # skips the selection grid, k-sensitivity and SHAP
"""

import io
import os
import re
import sys
import glob
import json
import base64
import argparse
import warnings
import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, f1_score,
                             adjusted_rand_score,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)

# =============================================================================
# 1. SETTINGS — the only block you normally edit
# =============================================================================

EXTRACT_DIR = "/mnt/HC_Volume_106672325/parq_b_data_extracted"
OUT_DIR     = "article_outputs"

# optional inputs; leave as None (or a wrong path) and the script simply skips them
GEOJSON_L3  = "viz_dashboard_data2/l3.geojson"   # for Figure 6 (map); needs geopandas
GEOJSON_KEY = "BUDGET_CODE"                       # geojson property holding the budget code
POLAND_PIT  = None   # e.g. "poland/pit_powiaty.csv"  : columns Year, Powiat, PIT
POLAND_PESEL = None  # e.g. "poland/pesel_ukr.csv"    : columns Year, Powiat, Registered

N_PC            = 10                 # principal components used everywhere
K               = 10                 # clusters in the main specification
K_SENS          = [8, 10, 12]        # Table A.3 — sensitivity to k (Reviewer 1)
N_PC_HIGH       = 50                 # second curve in Figure A.1
SEED            = 42
FORECAST_YEARS  = [2021, 2022, 2023, 2024, 2025]
STABLE_LABELS   = True               # renumber clusters 1..K by financial score (see §4)

# Subtract each item's year mean before PCA. Without this the clustering mostly
# separates YEARS: in the uncentred run one transition dominates every oblast in
# every period (6->8, then 8->9, then 7->5) and 997 of 1419 targets in 2025 fall
# in a cluster that never existed in training. Centred, a cluster means "how this
# community differs from the others in the same year".
CENTER_BY_YEAR  = True
# Drop items that are near-empty everywhere: they survive standardisation as
# full-weight noise directions and flatten the eigenvalue spectrum. 0 disables.
MIN_MEAN_SHARE  = 0.0005   # keep items averaging at least 0.05% of a budget
MIN_PRESENCE    = 0.02     # ... and non-zero in at least 2% of community-years
# On the real data this keeps 63 of 271 revenue items, which still carry 98.3% of
# the money, and lifts the variance held by 10 components from 11.4% to 36.3%.
# Set both to 0.0 to go back to every item.

# The reporting grid printed at the start of every run (skipped by --quick), and
# the years on which choosing a configuration is allowed. The forecast years after
# SELECT_YEARS are never used for the choice, so they stay a true out-of-sample test.
PC_GRID      = [10, 20, 50, 100]
K_GRID       = [5, 10, 15]
SELECT_YEARS = [2022, 2023]
TRIM_GRID    = [(0.0, 0.0), (0.0005, 0.02), (0.001, 0.05), (0.002, 0.10)]

#   code_col / value_col may be a single name or an ordered list of candidates; the
#   first one present in the file wins.
#
#   NOTE 1 — the doubling of program expenditures. A program_expenses file carries three
#   classifications at once (PK x FK x EK) and holds BOTH the detail rows and the per-
#   programme subtotals, which are the rows with COD_CONS_EK = 0 (their COD_CONS_EK_NAME
#   is empty). Summing every row therefore counts each hryvnia exactly twice, which is the
#   "prog / econ = 2.001 in all eight years" seen in the first run. keep_rows below drops
#   the subtotal rows; this is the same filter the earlier test_p.py applied.
#
#   NOTE 2 — functional expenditures. The functional_expenses folder has NO column called
#   FAKT_AMT (the outturn arrives as a duplicated header that pandas renames to
#   PLANS_AMT.1), so a naive loader silently reads the APPROVED PLAN. The functional
#   breakdown is also present in the program files, as COD_CONS_MB_FK, with a proper
#   FAKT_AMT next to it — so func is read from there: same rows as prog, grouped by FK,
#   which makes func and econ agree exactly instead of approximately.
EK_DETAIL_ROWS = lambda d: pd.to_numeric(d["COD_CONS_EK"], errors="coerce").fillna(0) > 0

CATEGORIES = {
    "income": {"folder": "empty_incomes",       "code_col": "COD_INCO",         "width": 8,
               "value_col": ["FAKT_AMT"], "name_col": "NAME_INC"},
    "prog":   {"folder": "program_expenses",    "code_col": "COD_CONS_MB_PK",   "width": 4,
               "value_col": ["FAKT_AMT"], "name_col": "COD_CONS_MB_PK_NAME",
               "needs": ["COD_CONS_EK"], "keep_rows": EK_DETAIL_ROWS},
    "econ":   {"folder": "economic_expenses",   "code_col": "COD_CONS_EK",      "width": 4,
               "value_col": ["FAKT_AMT"], "name_col": "COD_CONS_EK_NAME"},
    "func":   {"folder": "program_expenses",    "code_col": "COD_CONS_MB_FK",   "width": 4,
               "value_col": ["FAKT_AMT"], "name_col": "COD_CONS_MB_FK_NAME",
               "needs": ["COD_CONS_EK"], "keep_rows": EK_DETAIL_ROWS},
}
# columns that hold a plan rather than an outturn; using one is allowed but never silent
PLAN_COLS = {"ZAT_AMT", "PLANS_AMT"}
CAT_LABEL = {"income": "own revenues", "prog": "program expenditures",
             "econ": "expenditures (economic classification)",
             "func": "expenditures (functional classification)"}

# which categories go through PCA + clustering + forecasting (they need expert weights
# below for the stable numbering and for the score); the rest are loaded for Table 1 only
CLUSTER_CATS  = ["income", "prog"]
TABLE1_PANELS = ["econ", "income"]      # order of the panels in Table 1

def cat_suffix(cat):
    """First clustered category keeps the plain number, the next ones get b, c, ..."""
    i = CLUSTER_CATS.index(cat) if cat in CLUSTER_CATS else 0
    return "" if i == 0 else chr(ord("a") + i)

EXPERT_WEIGHTS = {
    "income": {"11010400": 1.0, "18050400": 1.0, "18050500": 1.0,
               "18010600": 0.8, "18010900": 0.8, "21081100": 0.5,
               "41033900": -0.5, "41020100": -1.0, "41030600": -1.0,
               "41030800": -1.0, "41050300": -1.0, "41050700": -1.0},
    "prog":   {"7365": 1.0, "7362": 1.0, "1184": 1.0, "7413": 0.5,
               "1021": 0.0, "1091": 0.0, "2010": 0.0, "0160": 0.0,
               "8600": -0.5, "7692": -1.0, "9220": -1.0, "9230": -1.0},
}

# items needed to test the PIT / education-subvention explanation (RQ3)
KEY_ITEMS = {
    "11010100": "PIT on wages",
    "11010200": 'PIT on military pay ("military PIT")',
    "11010400": "PIT on other income",
    "14040100": "Retail excise on tobacco",
    "18050400": "Single tax (individuals)",
    "18050500": "Single tax (agricultural producers)",
    "41020100": "Basic dotation",
    "41033900": "Educational subvention",
    "41036000": "Subvention: New Ukrainian School",
    "41036300": "Subvention: additional payments to teachers",
    "41035400": "Subvention: special educational needs",
}
KEY_GROUPS = {"PIT total (1101xxxx)": "1101", "Local taxes and fees (18xxxxxx)": "18",
              "Excise (14xxxxxx)": "14", "Transfers (41xxxxxx)": "41"}

# names used as labels in the tables — a superset of KEY_ITEMS, which stays short
# because it also decides which items RQ3 tracks
ITEM_NAMES = dict(KEY_ITEMS, **{
    "11010100": "Personal income tax (from wages)",
    "11010400": "Personal income tax (non-wage income)",
    "14040100": "Excise tax on tobacco and electronic cigarettes",
    "18010600": "Land tax (legal entities)",
    "18010900": "Land rent (individuals)",
    "18050400": "Single tax (individuals)",
    "18050500": "Single tax from agricultural producers",
    "21081100": "Administrative fines and other sanctions",
    "41020100": "Basic dotation",
    "41030600": "Subvention for district council expenses",
    "41030800": "Subvention for housing and utility subsidies",
    "41031000": "Subvention for solid fuel and gas subsidies",
    "41033900": "Educational subvention",
    "41035400": "Subvention for special educational needs",
    "41035800": "Subvention for war veterans support services",
    "41036000": 'Subvention for "New Ukrainian School"',
    "41036300": "Subvention for additional payments to teachers",
    "41050300": "Education subvention",
    "41050700": "Other subventions",
})
PROG_ITEMS = {
    "0160": "Leadership and management", "1021": "Secondary education",
    "1091": "Vocational education", "1184": 'Investment (New Ukrainian School)',
    "2010": "Inpatient medical care", "7362": "Investment (other)",
    "7365": "Investment (medical)", "7413": "Motor transport",
    "8600": "Debt servicing", "7692": "State guarantee servicing",
    "9220": "Subvention for benefits", "9230": "Other subventions",
}
TOP_ITEMS = 4          # budget items listed per cluster in Table A.1
EDU_2025_CODES = ["41036300", "41036000"]
MILITARY_PIT   = ["11010200"]

REGION_MAP = {
    "01": "Krym", "02": "Vinnytska", "03": "Volynska", "04": "Dnipropetrovska",
    "05": "Donetska", "06": "Zhytomyrska", "07": "Zakarpatska", "08": "Zaporizka",
    "09": "Ivano-Frankivska", "10": "Kyivska", "11": "Kirovohradska", "12": "Luhanska",
    "13": "Lvivska", "14": "Mykolaivska", "15": "Odeska", "16": "Poltavska",
    "17": "Rivnenska", "18": "Sumska", "19": "Ternopilska", "20": "Kharkivska",
    "21": "Khersonska", "22": "Khmelnytska", "23": "Cherkaska", "24": "Chernivetska",
    "25": "Chernihivska", "26": "Kyiv_City", "27": "Sevastopol",
}

BLUE, GREY, RED, DARK = "#2a78d6", "#c3c2b7", "#e34948", "#52514e"

# =============================================================================
# 2. PLUMBING — logging, saving, and the report collector
# =============================================================================

ap = argparse.ArgumentParser()
ap.add_argument("--reload", action="store_true", help="ignore the cache, re-read parquet")
ap.add_argument("--quick", action="store_true", help="skip k-sensitivity and SHAP")
ARGS = ap.parse_args()

os.makedirs(OUT_DIR, exist_ok=True)
EXTRA_DIR = os.path.join(OUT_DIR, "extra")        # working files, not part of the paper
os.makedirs(EXTRA_DIR, exist_ok=True)
LOG = open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8")

# Exactly these references become files in OUT_DIR — one file per number in the
# manuscript, nothing else. Everything else the script computes still appears in
# report.html and summary.txt, and its data goes to OUT_DIR/extra/.
MANUSCRIPT_REFS = ({f"Table {i}" for i in range(1, 6)}
                   | {f"Table A.{i}" for i in range(1, 4)}
                   | {f"Figure {i}" for i in range(1, 11)}
                   | {"Figure A.1", "Figure B.1", "Figure B.2"})

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

def log_df(title, df, max_rows=80):
    log("\n### " + title)
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 60,
                           "display.width", 220):
        log(df.to_string())

# report.html is assembled from this list as the script runs
REPORT = []   # dicts: kind=table|figure|note, ref, title, note, path, html

def extra_csv(df, name, index=True):
    """Working data that is not a numbered table: kept out of the main folder."""
    df.to_csv(os.path.join(EXTRA_DIR, name), index=index)

def write_bytes(path, blob, what):
    """Write in one call, then read the size back. A file that ends up empty is
    reported here instead of being discovered later in a file manager."""
    with open(path, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    size = os.path.getsize(path)
    if size == 0:
        log(f"   ERROR {what}: wrote 0 bytes to {path} — the disk is full or the path is "
            f"not writable. Check `df -h` on the output volume.")
    return size

def add_table(df, ref, title, name=None, note="", index=False, float_fmt="{:,.2f}"):
    """Numbered tables become Table_N.csv/.xlsx in OUT_DIR; anything else is kept
    in extra/ and still shown in report.html."""
    name = name or ref.replace(".", "_").replace(" ", "_")
    main = ref in MANUSCRIPT_REFS
    folder = OUT_DIR if main else EXTRA_DIR
    df.to_csv(os.path.join(folder, name + ".csv"), index=index)
    try:
        with pd.ExcelWriter(os.path.join(folder, name + ".xlsx")) as xw:
            df.to_excel(xw, sheet_name=ref[:31], index=index)
    except Exception as e:
        log(f"   (xlsx not written for {name}: {e}; `pip install openpyxl` to get it)")
    REPORT.append(dict(kind="table", ref=ref, title=title, note=note,
                       html=df.to_html(index=index, border=0, justify="left",
                                       float_format=lambda v: float_fmt.format(v)),
                       path=("" if main else "extra/") + name + ".csv"))
    log_df(f"{ref}. {title}", df)

def add_figure(fig, ref, title, name=None, note="", dpi=200):
    """Render to memory first, then write the bytes once and verify the size, so a
    truncated or empty .png cannot pass unnoticed."""
    name = name or ref.replace(".", "_").replace(" ", "_")
    main = ref in MANUSCRIPT_REFS
    folder = OUT_DIR if main else EXTRA_DIR
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    except Exception as e:                      # never lose the rest of the run
        plt.close(fig)
        log(f"   ERROR {ref}: figure could not be rendered ({e})")
        add_note(ref, title, f"figure could not be rendered: {e}")
        return
    plt.close(fig)
    blob = buf.getvalue()
    size = write_bytes(os.path.join(folder, name + ".png"), blob, ref)
    REPORT.append(dict(kind="figure", ref=ref, title=title, note=note,
                       path=("" if main else "extra/") + name + ".png", html=""))
    log(f"[figure] {ref}. {title}  ->  {name}.png  ({size/1024:.0f} kB)")

def add_note(ref, title, text):
    REPORT.append(dict(kind="note", ref=ref, title=title, note=text, html="", path=""))
    log(f"[note] {ref}: {text}")

def style_axes(ax):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return ax

def heatmap(ax, mat, fmt="{:.2f}", vmin=0, vmax=1, cmap="Blues", ylabels=None):
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=0)
    ax.set_yticks(range(mat.shape[0]), ylabels if ylabels is not None else mat.index)
    hi = vmax if vmax is not None else np.nanmax(mat.values)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * hi else "#0b0b0b")
    return im

log("run_all.py —", datetime.datetime.now().isoformat(timespec="seconds"))
log("versions: pandas", pd.__version__, "| sklearn", __import__("sklearn").__version__,
    "| xgboost", xgb.__version__, "| shap", shap.__version__, "| numpy", np.__version__)
log(f"settings: N_PC={N_PC} K={K} SEED={SEED} STABLE_LABELS={STABLE_LABELS} quick={ARGS.quick}")

# =============================================================================
# 3. LOADING
# =============================================================================

def norm_codes(series, width, cat):
    num = pd.to_numeric(series, errors="coerce")
    ok = num.notna()
    codes = num[ok].astype("int64").astype(str)
    codes = codes.str.zfill(7).str[-4:] if cat == "prog" else codes.str.zfill(width)
    out = pd.Series(np.nan, index=series.index, dtype=object)
    out[ok] = codes
    return out

def parse_month(s):
    """Month 1..12 out of REP_PERIOD in any format the OpenBudget files use."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.month.astype(float)
    st = s.astype(str).str.strip()
    month = pd.Series(np.nan, index=s.index)
    patterns = [r"^\d{1,2}\.(\d{1,2})\.\d{4}",   # 01.03.2020
                r"^(\d{1,2})[./-]\d{4}",          # 3.2020
                r"^\d{4}[-./](\d{1,2})",          # 2020-03
                r"^20\d{2}(0[1-9]|1[0-2])$",      # 202003
                r"^(\d{1,2})20\d{2}$",            # 32020
                r"^(\d{1,2})\.\d{1,3}$"]          # 3.202
    for pat in patterns:
        if month.notna().all():
            break
        month = month.fillna(pd.to_numeric(st.str.extract(pat)[0], errors="coerce"))
    month[(month < 1) | (month > 12)] = np.nan
    return month

def resolve_code_col(columns, config, cat):
    """The budget-code column is named differently per category and per vintage."""
    wanted = config["code_col"]
    for c in ([wanted] if isinstance(wanted, str) else wanted):
        if c in columns:
            return c
    guess = [c for c in columns
             if c != "COD_BUDGET" and any(t in c.upper()
                                          for t in ("COD", "PROG", "EK", "FK", "INC"))]
    if guess:
        log(f"   {cat}: column {wanted} absent, using {guess[0]} instead")
        return guess[0]
    return None

def resolve_value_col(columns, config, cat):
    """Pick the actual-spending column. Never fall back to a plan column silently:
    functional_expenses has no FAKT_AMT, and reading ZAT_AMT there would quietly
    substitute the approved budget for the outturn."""
    for c in config.get("value_col", ["FAKT_AMT"]):
        if c in columns:
            if c != "FAKT_AMT":
                log(f"   {cat}: NOTE no FAKT_AMT in these files; using '{c}', which sits in "
                    f"the position FAKT_AMT occupies in the other categories "
                    f"(duplicated header renamed by pandas). Verify with the cross-check below.")
            return c
    for c in columns:
        if c.upper().startswith("FAKT") and "V2MB" not in c.upper():
            log(f"   {cat}: amount column guessed as '{c}'")
            return c
    for c in ("PLANS_AMT", "ZAT_AMT"):
        if c in columns:
            log(f"   {cat}: WARNING only plan columns are present, falling back to '{c}'. "
                f"These are PLANNED, not actual, amounts — do not mix them with other "
                f"categories without saying so in the paper.")
            return c
    return None

def load_category(cat, config):
    files = glob.glob(f"{EXTRACT_DIR}/**/*{config['folder']}*.parquet", recursive=True)
    log(f"\n================ LOADING {cat.upper()}: {len(files)} files ================")
    if not files:
        log("   no files found — check EXTRACT_DIR")
        return None, {}, {}

    levels = pd.Series([("l" + m.group(2)) if (m := re.search(r"(^|[/\\])([123])_", f))
                        else "unknown" for f in files])
    log("files by level:", levels.value_counts().to_dict())
    keep_l3 = (levels == "l3").any()
    if keep_l3:
        log("-> keeping level-3 (hromada) files only")

    names, parts = {}, []
    fund_counts, tcs_ratios, short_years, bad_months = {}, [], {}, {}
    dropped_zero = [0, 0]      # summary rows removed / rows seen
    no_code = {}               # year -> [rows with no code, rows seen]
    need_cols = config.get("needs") or []        # extra columns keep_rows looks at
    name_col = config.get("name_col")            # official item name, straight from the file
    item_names = {}
    dropped_sub = [0, 0]       # subtotal rows removed by keep_rows / rows seen
    nondecr = [0, 0]
    base_cols = ["REP_PERIOD", "FUND_TYP", "COD_BUDGET"]
    code_col = val_col = None    # both resolved from the first file that opens
    logged_cols, n_done = False, 0

    for f, lvl in zip(files, levels):
        if keep_l3 and lvl != "l3":
            continue
        ym = re.search(r"_(\d{4})\.parquet", os.path.basename(f))
        if not ym:
            continue
        year = int(ym.group(1))
        for nm in re.findall(r"([^/\\]+)_(\d{10})", f):
            names[nm[1]] = nm[0]
        path_codes = re.findall(r"_(\d{10})(?=[/\\])", f)
        if not path_codes:
            continue
        bcode = path_codes[-1]

        try:
            df = pd.read_parquet(f, columns=base_cols + [code_col, val_col]
                                 + [c for c in need_cols]
                                 + ([name_col] if name_col else [])) \
                 if (code_col and val_col) else pd.read_parquet(f)
        except Exception:
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
        if code_col is None:
            code_col = resolve_code_col(list(df.columns), config, cat)
            if code_col is None:
                log(f"   {cat}: no budget-code column found in {f} — category skipped")
                return None, names, item_names
            log(f"   {cat}: budget-code column = {code_col}")
        if val_col is None:
            val_col = resolve_value_col(list(df.columns), config, cat)
            if val_col is None:
                log(f"   {cat}: no amount column found in {f} — category skipped")
                return None, names, item_names
        if not logged_cols:
            log("example file:", f)
            log("columns:", list(df.columns))
            log(f"   {cat}: amount column = {val_col}"
                + ("  <-- PLAN, not outturn" if val_col in PLAN_COLS else ""))
            logged_cols = True
        n_done += 1
        if n_done % 1000 == 0:
            print(f"   ... {n_done} files read", flush=True)
        if df.empty or code_col not in df.columns:
            continue
        if val_col not in df.columns:
            continue

        t = pd.DataFrame({
            "BUDGET_CODE": bcode,
            "Code": norm_codes(df[code_col], config["width"], cat),
            "Amount": pd.to_numeric(df[val_col], errors="coerce").fillna(0) / 1e6,
        })
        if "REP_PERIOD" in df.columns:
            mth = parse_month(df["REP_PERIOD"])
            if mth.isna().any() and year not in bad_months:
                bad_months[year] = df.loc[mth.isna(), "REP_PERIOD"].astype(str).unique()[:5].tolist()
            t["Month"] = mth.fillna(0)
        else:
            t["Month"] = 0
        t["Fund"] = (df["FUND_TYP"].astype(str).str.upper().str.strip()
                     if "FUND_TYP" in df.columns else "ALL")
        # official names of the budget items, taken from the files rather than typed by
        # hand; collected until every code has one, then the lookup stops costing anything
        if name_col and name_col in df.columns and len(item_names) < 4000:
            pair = pd.DataFrame({"Code": t["Code"].values,
                                 "Name": df[name_col].astype(str).values}).dropna()
            pair = pair[pair["Name"].str.strip().ne("") & ~pair["Name"].isin(["nan", "None"])]
            if not pair.empty:
                for cd, nm in pair.groupby("Code")["Name"].first().items():
                    item_names.setdefault(cd, nm.strip())
        for c in need_cols:                 # columns keep_rows needs, dropped again below
            if c in df.columns:
                t["need:" + c] = df[c].values
        del df

        nc = no_code.setdefault(year, [0, 0])
        nc[0] += int(t["Code"].isna().sum())
        nc[1] += len(t)
        t = t.dropna(subset=["Code"])
        if cat in ("econ", "func"):
            # code 0 is the structural summary row; keeping it double counts the whole budget
            before = len(t)
            t = t[t["Code"].astype(int) > 0]
            dropped_zero[0] += before - len(t)
            dropped_zero[1] += before
        if t.empty:
            continue

        for k_, v_ in t["Fund"].value_counts().items():
            fund_counts[k_] = fund_counts.get(k_, 0) + int(v_)
        if (t["Fund"] == "T").any():
            tot_t = t.loc[t["Fund"] == "T", "Amount"].sum()
            tot_o = t.loc[t["Fund"] != "T", "Amount"].sum()
            if tot_t > 0 and tot_o > 0:
                tcs_ratios.append(tot_o / tot_t)
            t = t[t["Fund"] == "T"]

        keep = config.get("keep_rows")
        if keep is not None and all("need:" + c in t.columns for c in need_cols):
            before = len(t)
            t = t[keep(t.rename(columns={"need:" + c: c for c in need_cols}))]
            dropped_sub[0] += before - len(t)
            dropped_sub[1] += before
        t = t[[c for c in t.columns if not c.startswith("need:")]]
        if t.empty:
            continue

        g = (t.groupby(["BUDGET_CODE", "Code", "Month"])["Amount"].sum().reset_index()
               .sort_values(["BUDGET_CODE", "Code", "Month"]))
        d = g.groupby(["BUDGET_CODE", "Code"])["Amount"].diff().dropna()
        nondecr[0] += int((d >= -1e-9).sum())
        nondecr[1] += int(len(d))

        last_month = g.groupby(["BUDGET_CODE", "Code"])["Month"].transform("max")
        if g["Month"].max() < 12:
            short_years[year] = short_years.get(year, 0) + 1
        a_last = g[g["Month"] == last_month].groupby(["BUDGET_CODE", "Code"])["Amount"].sum()
        a_sum = g.groupby(["BUDGET_CODE", "Code"])["Amount"].sum()
        part = pd.concat([a_last.rename("A_last"), a_sum.rename("A_sum")], axis=1).reset_index()
        part["Year"] = year
        parts.append(part)

    if not parts:
        return None, names, item_names

    if dropped_zero[1]:
        log(f"summary rows (code 0) removed: {dropped_zero[0]} of {dropped_zero[1]} "
            f"({100 * dropped_zero[0] / dropped_zero[1]:.1f}%)")
    bad_code = {y: round(100 * v[0] / v[1], 1) for y, v in sorted(no_code.items())
                if v[1] and v[0] / v[1] > 0.01}
    if bad_code:
        log(f"rows with an empty '{code_col}' (dropped), % per year: {bad_code}. A year with a "
            f"high share here is under-read: that classification was simply not filled in the "
            f"source files for those years.")
    if dropped_sub[1]:
        log(f"per-programme subtotal rows (COD_CONS_EK = 0) removed: {dropped_sub[0]} of "
            f"{dropped_sub[1]} ({100 * dropped_sub[0] / dropped_sub[1]:.1f}%). Without this "
            f"filter every hryvnia is counted twice — see NOTE 1 in the settings block.")
    log("FUND_TYP values:", fund_counts)
    if tcs_ratios:
        log(f"median (C+S)/T per file = {np.median(tcs_ratios):.3f} -> keeping T only")
    allp = pd.concat(parts, ignore_index=True)
    del parts

    frac = nondecr[0] / nondecr[1] if nondecr[1] else np.nan
    cumulative = bool(frac >= 0.8)
    log(f"non-decreasing month steps = {frac:.2f} -> "
        + ("CUMULATIVE reports: last month of each year taken"
           if cumulative else "monthly flows: months summed"))
    if short_years:
        log("files ending before December (year: count):", short_years)
    if bad_months:
        log("WARNING unparsed REP_PERIOD (year: examples):", bad_months)

    allp["Amount"] = allp["A_last"] if cumulative else allp["A_sum"]
    annual = allp.groupby(["BUDGET_CODE", "Code", "Year"])["Amount"].sum().reset_index()
    del allp

    codes = annual["Code"].unique()
    parents = ([c for c in codes if any((o != c) and o.startswith(c.rstrip("0")) for o in codes)]
               if len(codes) < 3000 else [])
    if parents:
        tot_par = annual[annual["Code"].isin(parents)]["Amount"].sum()
        leaf = annual[~annual["Code"].isin(parents)]["Amount"].sum()
        log(f"parent (summary) codes: {len(parents)}; their share = "
            f"{tot_par / annual['Amount'].sum():.2f}")
        if tot_par > 0.3 * leaf:
            annual = annual[~annual["Code"].isin(parents)]
            log("-> parent codes dropped (they are sums of sub-codes)")

    annual = annual[annual["Amount"] > 0]
    log("budgets per year:", annual.groupby("Year")["BUDGET_CODE"].nunique().to_dict())

    if item_names:
        log(f"   item names read from the files: {len(item_names)} codes")
    return annual, names, item_names


DATA, NAMES, DATA_ITEM_NAMES = {}, {}, {}
for cat, config in CATEGORIES.items():
    cache_f = os.path.join(OUT_DIR, f"cache_{cat}.pkl")
    if os.path.exists(cache_f) and not ARGS.reload:
        cached = pd.read_pickle(cache_f)
        df_, nm = cached[0], cached[1]
        inm = cached[2] if len(cached) > 2 else {}   # caches written before this change
        log(f"{cat}: loaded from cache {cache_f}  (--reload to re-read parquet)"
            + ("" if inm else "  — no item names in this cache, run --reload once to add them"))
    else:
        df_, nm, inm = load_category(cat, config)
        if df_ is not None:
            pd.to_pickle((df_, nm, inm), cache_f)
    if df_ is not None:
        DATA[cat] = df_
        NAMES.update(nm)
        DATA_ITEM_NAMES[cat] = inm or {}

def item_label(code, cat=None):
    """Official name from the data; the hand-written list is only a fallback."""
    code = str(code)
    if cat and code in DATA_ITEM_NAMES.get(cat, {}):
        return DATA_ITEM_NAMES[cat][code]
    for d in DATA_ITEM_NAMES.values():
        if code in d:
            return d[code]
    return (PROG_ITEMS if cat == "prog" else ITEM_NAMES).get(code, "")

if not DATA:
    log("\nNo data loaded. Set EXTRACT_DIR correctly and re-run.")
    sys.exit(1)

def name_of(code):
    return NAMES.get(str(code), str(code))

# ---- sanity checks on what was actually read -------------------------------
log("\n================ SANITY CHECKS ON THE LOADED DATA ================")
chk = []
for cat, df_ in DATA.items():
    dup = df_.duplicated(["BUDGET_CODE", "Code", "Year"]).sum()
    g = df_.groupby("Year")
    chk.append(pd.DataFrame({
        "Category": cat,
        "Year": g["Amount"].sum().index,
        "Total, million UAH": g["Amount"].sum().round(0).values,
        "Communities": g["BUDGET_CODE"].nunique().values,
        "Distinct codes": g["Code"].nunique().values,
    }))
    if dup:
        log(f"WARNING {cat}: {dup} duplicated (community, code, year) rows — "
            f"something is being counted twice")
chk = pd.concat(chk, ignore_index=True)
log_df("totals by category and year (after filtering)", chk)

# econ, prog and func are three classifications of the SAME spending, so their annual
# totals have to agree. A category that is out of line was read or filtered wrongly.
exp_cats = [c for c in ["econ", "prog", "func"] if c in DATA]
if len(exp_cats) > 1:
    piv = chk[chk["Category"].isin(exp_cats)].pivot(
        index="Year", columns="Category", values="Total, million UAH")
    base = "econ" if "econ" in piv.columns else exp_cats[0]
    for c in piv.columns:
        if c != base:
            piv[f"{c} / {base}"] = (piv[c] / piv[base]).round(3)
    log_df(f"cross-check: the same expenditures under different classifications "
           f"(ratios should be close to 1.000)", piv)
    high = [c for c in piv.columns if "/" in c and (piv[c] > 1.1).any()]
    low = [c for c in piv.columns if "/" in c and (piv[c] < 0.9).any()]
    if high:
        log("WARNING above 1.1 in at least one year: " + ", ".join(high)
            + "  -> that classification is double counted: the files hold both detail rows "
              "and subtotals. Add a keep_rows filter for it (see NOTE 1 in the settings).")
    if low:
        log("NOTE below 0.9 in at least one year: " + ", ".join(low)
            + "  -> that classification is under-read, normally because its code column is "
              "empty in the source for those years — see the 'rows with an empty ...' line "
              "in the loading block above. It is not double counting.")
    if not high and not low:
        log("OK: the expenditure classifications agree within 10% in every year")
extra_csv(chk, "sanity_checks.csv", index=False)

# =============================================================================
# 4. PCA + CLUSTERING (one run per category, reused by everything below)
# =============================================================================

def shares_matrix(annual, drop_codes=None):
    a = annual if not drop_codes else annual[~annual["Code"].isin(drop_codes)]
    pv = a.pivot_table(index=["BUDGET_CODE", "Year"], columns="Code",
                       values="Amount", aggfunc="sum", fill_value=0)
    return pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

def trim_codes(shares, min_mean=None, min_presence=None):
    """Drop budget items that are near-empty everywhere. Standardisation gives such
    an item the same weight as PIT, so hundreds of them flatten the spectrum."""
    min_mean = MIN_MEAN_SHARE if min_mean is None else min_mean
    min_presence = MIN_PRESENCE if min_presence is None else min_presence
    if not min_mean and not min_presence:
        return shares, dict(kept=shares.shape[1], dropped=0, money=100.0)
    keep = ((shares.mean() >= min_mean) & ((shares > 0).mean() >= min_presence))
    if not keep.any():
        return shares, dict(kept=shares.shape[1], dropped=0, money=100.0)
    money = 100.0 * shares.loc[:, keep].sum().sum() / shares.sum().sum()
    return shares.loc[:, keep], dict(kept=int(keep.sum()),
                                     dropped=int((~keep).sum()), money=round(money, 2))

def center_by_year(shares):
    """Subtract each item's mean for its year, so what is left is how a community
    differs from the others in the SAME year rather than which year it is."""
    years = shares.index.get_level_values("Year")
    return shares - shares.groupby(years).transform("mean")

def prep_shares(annual, drop_codes=None, center=None, min_mean=None, min_presence=None):
    """Return (raw shares for interpretation, matrix that goes into PCA)."""
    raw = shares_matrix(annual, drop_codes)
    model, info = trim_codes(raw, min_mean, min_presence)
    center = CENTER_BY_YEAR if center is None else center
    if center:
        model = center_by_year(model)
    return raw, model, info

def broken_stick(p, n):
    """Share of variance the first n components would take if variance were split
    at random between p dimensions. A component below its threshold is not
    distinguishable from noise."""
    h = [sum(1.0 / i for i in range(k, p + 1)) / p for k in range(1, n + 1)]
    return 100 * h[0], 100 * sum(h)

def year_purity(pdf):
    """100% means every cluster sits in a single year, i.e. the clustering is just
    labelling years. Lower is better."""
    ct = pd.crosstab(pdf["Cluster"], pdf["Year"])
    w = ct.sum(axis=1)
    return float(100 * (ct.max(axis=1) / w * w).sum() / w.sum())

def pca_kmeans(shares, k=K, n_pc=N_PC, weights=None, label_shares=None):
    X = StandardScaler().fit_transform(shares)
    n_pc = min(n_pc, shares.shape[1], shares.shape[0])
    pca = PCA(n_components=n_pc, random_state=SEED)
    comp = pca.fit_transform(X)
    pc_cols = [f"pc{i+1}" for i in range(n_pc)]
    pdf = pd.DataFrame(comp, columns=pc_cols, index=shares.index).reset_index()
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    pdf["Cluster"] = km.fit_predict(pdf[pc_cols]) + 1

    # ---- stable, interpretable numbering: cluster 1 = most self-sufficient.
    # k-means labels are arbitrary, so without this they change whenever k,
    # the seed or the input changes, and tables stop matching figures.
    if STABLE_LABELS and weights:
        # the score is always read off the RAW shares: centred values are not shares
        lab = shares if label_shares is None else label_shares
        sh = lab.reset_index().merge(pdf[["BUDGET_CODE", "Year", "Cluster"]],
                                     on=["BUDGET_CODE", "Year"])
        cols = [c for c in lab.columns if c in weights]
        score = (sh[cols] * pd.Series({c: weights[c] for c in cols})).sum(axis=1)
        order = (score.groupby(sh["Cluster"]).mean()
                 .sort_values(ascending=False).index.tolist())
        mapping = {old: new for new, old in enumerate(order, start=1)}
        pdf["Cluster"] = pdf["Cluster"].map(mapping)
        log("stable cluster numbering (old -> new):",
            {int(o): int(n) for o, n in mapping.items()})

    pdf = pdf.sort_values(["BUDGET_CODE", "Year"]).reset_index(drop=True)
    pdf["Region"] = pdf["BUDGET_CODE"].astype(str).str[:2].map(REGION_MAP)
    return pdf, pca, X, pc_cols

def ml_frame(pdf, pc_cols, k):
    p = pdf.sort_values(["BUDGET_CODE", "Year"]).copy()
    g = p.groupby("BUDGET_CODE")
    for c in pc_cols:
        p["prev_" + c] = g[c].shift(1)
    p["Prev_Cluster"] = g["Cluster"].shift(1)
    p["Target"] = g["Cluster"].shift(-1)
    ok = (g["Year"].shift(1) == p["Year"] - 1) & (g["Year"].shift(-1) == p["Year"] + 1)
    p = p[ok].dropna(subset=["Prev_Cluster", "Target"]).copy()
    p["Target"] = p["Target"].astype(int)
    p["Prev_Cluster"] = p["Prev_Cluster"].astype(int)
    for c in range(1, k + 1):
        p[f"cl_{c}"] = (p["Cluster"] == c).astype(int)
        p[f"prevcl_{c}"] = (p["Prev_Cluster"] == c).astype(int)
    num = pc_cols + ["prev_" + c for c in pc_cols]
    dummies = [f"cl_{c}" for c in range(1, k + 1)] + [f"prevcl_{c}" for c in range(1, k + 1)]
    return p, num, dummies

def feature_group(f):
    if f.startswith("prevcl_"):
        return "cluster (t−1)"
    if f.startswith("cl_"):
        return "cluster (t)"
    return f


def transitions(pdf):
    p = pdf.sort_values(["BUDGET_CODE", "Year"]).copy()
    p["Prev_Cluster"] = p.groupby("BUDGET_CODE")["Cluster"].shift(1)
    p["Prev_Year"] = p.groupby("BUDGET_CODE")["Year"].shift(1)
    p = p[p["Prev_Year"] == p["Year"] - 1]
    p["Changed"] = p["Cluster"] != p["Prev_Cluster"]
    p["Period"] = (p["Year"] - 1).astype(int).astype(str) + "–" + p["Year"].astype(str)
    return p

# ---------------------------------------------------------------------------
# MODEL SELECTION — runs before everything else and only reports. It chooses
# nothing by itself: it prints what each (components, clusters) pair would give,
# measured on SELECT_YEARS only, so the later forecast years stay a clean test.
# ---------------------------------------------------------------------------

def quick_forecast(pdf, pc_cols, k, years):
    """XGBoost vs persistence, averaged over the selection years."""
    p, num, dummies = ml_frame(pdf, pc_cols, k)
    feats = num + dummies
    rows = []
    for T in years:
        train, test = p[p["Year"] <= T - 2], p[p["Year"] == T - 1]
        if train.empty or test.empty or train["Target"].nunique() < 2:
            continue
        y_tr, y_te = train["Target"].values, test["Target"].values
        le = LabelEncoder().fit(y_tr)
        xg = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               random_state=SEED, tree_method="hist", n_jobs=-1)
        xg.fit(train[feats], le.transform(y_tr))
        yp = le.inverse_transform(xg.predict(test[feats]).astype(int))
        pers = test["Cluster"].values
        rows.append([accuracy_score(y_te, yp), f1_score(y_te, yp, average="macro"),
                     accuracy_score(y_te, pers), f1_score(y_te, pers, average="macro")])
    return np.array(rows).mean(axis=0) if rows else np.full(4, np.nan)

if not ARGS.quick:
    log("\n================ MODEL SELECTION ================")
    log(f"grid: components {PC_GRID} x clusters {K_GRID}, centred and uncentred; "
        f"chosen on forecast years {SELECT_YEARS} only")

    for cat in CLUSTER_CATS:
        if cat not in DATA:
            continue
        raw = shares_matrix(DATA[cat])
        log(f"\n---- {cat}: shares matrix {raw.shape[0]} community-years x "
            f"{raw.shape[1]} budget items")

        # how far the tail of near-empty codes can be cut, and what it buys
        trim_rows = []
        for mm, mp in TRIM_GRID:
            sub, info = trim_codes(raw, mm, mp)
            n = min(10, sub.shape[1])
            ev = PCA(n_components=n, random_state=SEED).fit(
                StandardScaler().fit_transform(sub)).explained_variance_ratio_.sum()
            bs1, bs10 = broken_stick(sub.shape[1], n)
            trim_rows.append({"min mean share, %": round(100 * mm, 3),
                              "min presence, %": round(100 * mp, 1),
                              "items kept": info["kept"],
                              "money covered, %": info["money"],
                              "EV of 10 components, %": round(100 * ev, 1),
                              "broken-stick top-10, %": round(bs10, 1)})
        trim_df = pd.DataFrame(trim_rows)
        log_df(f"cutting the tail of near-empty items ({cat}): what it costs and "
               f"what it buys", trim_df)
        extra_csv(trim_df, f"code_trimming_{cat}.csv", index=False)

        rows = []
        for centred in (False, True):
            base, _ = trim_codes(raw)
            Xm = center_by_year(base) if centred else base
            Xs = StandardScaler().fit_transform(Xm)
            full_lab = {}                       # k-means on the full matrix, no PCA
            for k in K_GRID:
                full_lab[k] = KMeans(n_clusters=k, random_state=SEED,
                                     n_init=10).fit_predict(Xs)
            seen_npc = set()
            for n_pc in PC_GRID:
                npc = min(n_pc, Xm.shape[1], Xm.shape[0])
                if npc in seen_npc:     # the grid asks for more components than
                    continue            # the matrix has: same run, skip the repeat
                seen_npc.add(npc)
                pca_s = PCA(n_components=npc, random_state=SEED).fit(Xs)
                Z = pca_s.transform(Xs)
                ev = 100 * pca_s.explained_variance_ratio_.sum()
                bs1, bs10 = broken_stick(Xm.shape[1], npc)
                for k in K_GRID:
                    lab = KMeans(n_clusters=k, random_state=SEED,
                                 n_init=10).fit_predict(Z)
                    seeds = [KMeans(n_clusters=k, random_state=sd,
                                    n_init=10).fit_predict(Z) for sd in (1, 7)]
                    ari_seed = np.mean([adjusted_rand_score(lab, o) for o in seeds])
                    ari_full = adjusted_rand_score(lab, full_lab[k])
                    pdf_ = pd.DataFrame(Z, columns=[f"pc{i+1}" for i in range(npc)],
                                        index=Xm.index).reset_index()
                    pdf_["Cluster"] = lab + 1
                    sil = silhouette_score(Z, lab, sample_size=min(5000, len(Z)),
                                           random_state=SEED)
                    acc, f1m, pacc, pf1 = quick_forecast(
                        pdf_, [f"pc{i+1}" for i in range(npc)], k, SELECT_YEARS)
                    rows.append({
                        "centred": "yes" if centred else "no",
                        "components": npc, "clusters": k,
                        "EV, %": round(ev, 1),
                        "broken-stick, %": round(bs10, 1),
                        "silhouette": round(float(sil), 3),
                        "ARI seed": round(float(ari_seed), 3),
                        "ARI vs full matrix": round(float(ari_full), 3),
                        "year purity, %": round(year_purity(pdf_), 1),
                        "accuracy": round(float(acc), 3),
                        "F1-macro": round(float(f1m), 3),
                        "persistence accuracy": round(float(pacc), 3),
                        "persistence F1": round(float(pf1), 3)})
                    log(f"   {cat} centred={centred} pc={npc:3d} k={k:2d} "
                        f"EV={ev:5.1f}% sil={sil:.3f} ARIseed={ari_seed:.3f} "
                        f"ARIfull={ari_full:.3f} year={rows[-1]['year purity, %']:5.1f}% "
                        f"acc={acc:.3f} (persist {pacc:.3f})")
        sel = pd.DataFrame(rows)
        log_df(f"model selection grid ({cat})", sel, max_rows=200)
        extra_csv(sel, f"model_selection_{cat}.csv", index=False)

    log("\nHow to read this:")
    log("  year purity   — 100% means the clusters are just year labels. This is the")
    log("                  number centring is meant to bring down.")
    log("  ARI seed      — below ~0.5 the partition is not reproducible; ignore the")
    log("                  rest of that row.")
    log("  ARI vs full   — high means PCA does not change who groups with whom, so a")
    log("                  low explained variance costs nothing.")
    log("  EV vs broken-stick — EV below the broken-stick column means the components")
    log("                  do not stand out from a random split of variance.")
    log("  accuracy      — compare with the persistence column, not with 1.0.")
    log(f"\nThe run now continues with N_PC={N_PC}, K={K}, CENTER_BY_YEAR={CENTER_BY_YEAR}. "
        f"If another row looks better, change those three at the top and run again; "
        f"the grid itself is skipped by --quick.")

RES = {}
for cat in CLUSTER_CATS:
    if cat not in DATA:
        continue
    log(f"\n================ PCA / CLUSTERING: {cat.upper()} ================")
    shares, model_shares, trim_info = prep_shares(DATA[cat])
    log(f"   items: {trim_info['kept']} kept, {trim_info['dropped']} dropped, "
        f"{trim_info['money']}% of the money covered"
        + ("; centred within each year" if CENTER_BY_YEAR else ""))
    if STABLE_LABELS and cat not in EXPERT_WEIGHTS:
        log(f"   note: no expert weights for '{cat}', so its cluster numbers stay "
            f"as k-means produced them")
    pdf, pca, X, pc_cols = pca_kmeans(model_shares, weights=EXPERT_WEIGHTS.get(cat),
                                      label_shares=shares)
    ch = transitions(pdf)
    RES[cat] = dict(shares=shares, model_shares=model_shares, pdf=pdf, pca=pca,
                    X=X, pc_cols=pc_cols, ch=ch)
    log(f"   year purity of the clusters: {year_purity(pdf):.1f}% "
        f"(100% would mean the clusters are just year labels)")

    log("explained variance pc1..pcN (%):",
        np.round(pca.explained_variance_ratio_ * 100, 1).tolist(),
        "| cumulative:", round(pca.explained_variance_ratio_.sum() * 100, 1))
    log("clusters per year:")
    log(pd.crosstab(pdf["Year"], pdf["Cluster"]).to_string())

    out = pdf.copy()
    out["Name"] = out["BUDGET_CODE"].map(name_of)
    extra_csv(out, f"clusters_{cat}_all_years.csv", index=False)
    extra_csv(out[out["Year"] == out["Year"].max()][["BUDGET_CODE", "Name", "Region", "Cluster"]],
              f"clusters_{cat}_last_year_for_map.csv", index=False)

# =============================================================================
# 5. TABLE 1 — descriptive statistics
# =============================================================================

T1_COLS = ["Year", "N", "Mean", "SD", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max"]
panels = []
for panel, cat in zip("ABCD", TABLE1_PANELS):
    if cat not in DATA:
        continue
    tot = DATA[cat].groupby(["Year", "BUDGET_CODE"])["Amount"].sum().reset_index()
    t = tot.groupby("Year")["Amount"].describe()[["count", "mean", "std", "min",
                                                  "25%", "50%", "75%", "max"]]
    log_df(f"Table 1, panel {panel} — total across all budgets ({cat}), million UAH",
           tot.groupby("Year")["Amount"].sum().round(0).to_frame("Total, all budgets"))
    t.columns = T1_COLS[1:]
    t = t.round(1).reset_index().rename(columns={"index": "Year"})
    t["Year"] = t["Year"].astype(int)
    t["N"] = t["N"].astype(int)
    head = pd.DataFrame([{c: ("" if c != "Year" else
                              f"Panel {panel}. {CAT_LABEL[cat].capitalize()}, million UAH")
                          for c in T1_COLS}])
    blank = pd.DataFrame([{c: "" for c in T1_COLS}])
    panels += ([blank] if panels else []) + [head, t[T1_COLS]]

if panels:
    add_table(pd.concat(panels, ignore_index=True), "Table 1",
              "Descriptive statistics of annual municipal budgets, million UAH",
              note="N is the number of budget units reporting in that year — it rises with the "
                   "amalgamation reform, so the panel is unbalanced and year-on-year means are "
                   "not directly comparable. The totals across all budgets, which can be checked "
                   "against the official consolidated figures, are in summary.txt.")

# =============================================================================
# 6. FIGURE 1 — the pipeline
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 4))
ax.axis("off")
W, H = 0.26, 0.22
boxes = {
    "Budget data\n2018–2025":          (0.03, 0.60),
    "Dimensionality\nreduction (PCA)": (0.37, 0.60),
    "Clustering\n(k-means)":           (0.71, 0.60),
    "Forecasting\n(XGBoost)":          (0.42, 0.14),
    "Transitions\nand scoring":        (0.72, 0.14),
}
for txt, (x, y) in boxes.items():
    ax.add_patch(plt.Rectangle((x, y), W, H, transform=ax.transAxes,
                               facecolor="#dbe7f6", edgecolor=DARK, lw=1.2,
                               joinstyle="round"))
    ax.text(x + W / 2, y + H / 2, txt, transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5)
arrows = [((0.29, 0.71), (0.37, 0.71)), ((0.63, 0.71), (0.71, 0.71)),
          ((0.78, 0.60), (0.58, 0.37)), ((0.88, 0.60), (0.86, 0.37))]
for (x1, y1), (x2, y2) in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3))
add_figure(fig, "Figure 1", "The analytical pipeline", name="fig1_pipeline",
           note="PCA compresses ~300 budget items into latent components, k-means groups "
                "communities into fiscal profiles, XGBoost forecasts next-year profiles, and "
                "transition analysis and scoring summarise structural change.")

# =============================================================================
# 7. FIGURE 2 + loadings — choice of k, and what the components mean
# =============================================================================

ks = list(range(2, 16))
ksel = {}
for cat in RES:
    pdf, pc_cols = RES[cat]["pdf"], RES[cat]["pc_cols"]
    Xp = pdf[pc_cols].values
    wcss, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xp)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(Xp, km.labels_,
                                    sample_size=min(5000, len(Xp)), random_state=SEED))
    kdf = pd.DataFrame({"k": ks, "WCSS": wcss, "Silhouette": sil})
    ksel[cat] = kdf
    extra_csv(kdf, f"k_selection_{cat}.csv", index=False)
    log_df(f"WCSS / silhouette by k ({cat})", kdf.round(3))

    # loadings: which budget items stand behind each component
    load = pd.DataFrame(RES[cat]["pca"].components_.T,
                        index=RES[cat]["model_shares"].columns, columns=pc_cols)
    RES[cat]["load"] = load
    rows = []
    for pc in pc_cols:
        top = load[pc].reindex(load[pc].abs().sort_values(ascending=False).index).head(8)
        for code, w in top.items():
            rows.append({"Component": pc, "Code": code,
                         "Item": KEY_ITEMS.get(code, ""), "Loading": round(w, 4)})
    tops = pd.DataFrame(rows)
    extra_csv(tops, f"pc_top_loadings_{cat}.csv", index=False)
    if cat == "income":
        add_table(tops[tops["Component"].isin(pc_cols[:3])], "Table A.4",
                  "Budget items with the largest loadings on the three leading components "
                  "(own revenues)", name="Table_A_4",
                  note="This is what links the SHAP results to actual budget items: SHAP ranks "
                       "components, and this table says which items each component is made of.")
        kl = load.reindex([c for c in KEY_ITEMS if c in load.index]).round(3)
        kl.insert(0, "Item", [KEY_ITEMS[c] for c in kl.index])
        add_table(kl.reset_index().rename(columns={"index": "Code"}), "Table A.5",
                  "Loadings of the key budget items (incl. military PIT) on every component",
                  name="Table_A_5")

# one Figure 2 for the whole paper: one column per clustered category
if ksel:
    cols2 = [c for c in CLUSTER_CATS if c in ksel]
    fig, axes = plt.subplots(2, len(cols2), figsize=(5.6 * len(cols2), 6),
                             sharex=True, squeeze=False)
    for j, cat in enumerate(cols2):
        kdf = ksel[cat]
        axes[0, j].plot(kdf["k"], kdf["WCSS"], marker="o", color=BLUE, lw=2)
        axes[1, j].plot(kdf["k"], kdf["Silhouette"], marker="s", color=RED, lw=2)
        axes[0, j].set_title(CAT_LABEL[cat].capitalize(), fontsize=11)
        axes[1, j].set_xlabel("Number of clusters (k)")
        for i in (0, 1):
            axes[i, j].axvline(K, color=DARK, ls="--", lw=1.2)
            style_axes(axes[i, j])
    axes[0, 0].set_ylabel("WCSS (inertia)")
    axes[1, 0].set_ylabel("Silhouette score")
    fig.suptitle(f"Choice of the number of clusters (dashed line: k = {K})", y=0.99)
    plt.tight_layout()
    add_figure(fig, "Figure 2", "Choice of the number of clusters",
               name="fig2_k_selection",
               note="Lower WCSS means more compact clusters; a higher silhouette means "
                    "better-separated profiles. One column per category, so the paper has a "
                    "single Figure 2 rather than one file per category.")

# =============================================================================
# 8. TABLE A.1 — cluster profiles from the actual shares
# =============================================================================

for cat in RES:
    shares, pdf = RES[cat]["shares"], RES[cat]["pdf"]

    sh = shares.reset_index().merge(pdf[["BUDGET_CODE", "Year", "Cluster"]],
                                    on=["BUDGET_CODE", "Year"])
    prof = sh.groupby("Cluster")[list(shares.columns)].mean()

    rows, detail = [], []
    for c in sorted(prof.index):
        top = prof.loc[c].sort_values(ascending=False).head(TOP_ITEMS)
        tot = float(top.sum())
        rows.append({"Code": f"cluster {int(c)}", "Share of impact (%)": None,
                     "Feature Name": ""})
        for code, v in top.items():
            rows.append({"Code": code,
                         "Share of impact (%)": round(100 * v / tot, 2) if tot else 0.0,
                         "Feature Name": item_label(code, cat)})
        yrs = pdf.loc[pdf["Cluster"] == c, "Year"]
        d = {"Cluster": int(c), "Observations": int((pdf["Cluster"] == c).sum()),
             "Years present": f"{yrs.min()}–{yrs.max()}" if len(yrs) else "",
             "Leading items (mean share of the budget, %)":
                 "; ".join(f"{k} {v*100:.1f}" for k, v in top.items())}
        if cat == "income":
            for g, pref in KEY_GROUPS.items():
                cols = [x for x in shares.columns if x.startswith(pref)]
                d[g] = round(float(sh.loc[sh["Cluster"] == c, cols].sum(axis=1).mean()
                                   * 100), 1) if cols else np.nan
        detail.append(d)

    tA1 = pd.DataFrame(rows)
    ref = "Table A.1" + cat_suffix(cat)
    add_table(tA1, ref, f"Dominant budget items by cluster — {CAT_LABEL[cat]}",
              name="Table_A_1" if cat == CLUSTER_CATS[0] else f"Table_A_1_{cat}",
              float_fmt="{:,.2f}",
              note="For every cluster, the four items with the largest mean share of the budget "
                   "across all its community-years; 'share of impact' renormalises those four to "
                   "100%, so the column says how the cluster's profile is split between them.")
    log_df(f"cluster profiles ({cat}) — observations, span and group shares",
           pd.DataFrame(detail))

# =============================================================================
# 9. TABLE A.2 + FIGURE 5 — transitions and regional dynamics
# =============================================================================

for cat in RES:
    pdf, ch = RES[cat]["pdf"], RES[cat]["ch"]

    nat = ch.groupby("Period")["Changed"].agg(["mean", "size"])
    nat.columns = ["Share of communities changing cluster, %", "Observations"]
    nat["Share of communities changing cluster, %"] = (
        nat["Share of communities changing cluster, %"] * 100).round(1)
    log_df(f"national transition rate ({cat})", nat)

    reg = ch.groupby(["Region", "Period"])["Changed"].mean().unstack() * 100
    RES[cat]["reg"] = reg
    extra_csv(reg.round(1), f"instability_by_oblast_{cat}.csv")

    tr = ch[ch["Changed"]].copy()
    tr["Transition"] = (tr["Prev_Cluster"].astype(int).astype(str) + "->"
                        + tr["Cluster"].astype(str))
    n_reg = ch.groupby(["Region", "Period"]).size()
    cnt = tr.groupby(["Region", "Period", "Transition"]).size().reset_index(name="n")
    cnt["pct"] = cnt.apply(lambda r: 100 * r["n"] / n_reg[(r["Region"], r["Period"])], axis=1)
    cnt = cnt.sort_values(["Region", "Period", "n"], ascending=[True, True, False])
    top3 = cnt.groupby(["Region", "Period"]).head(3).copy()
    top3["txt"] = top3["Transition"] + " (" + top3["pct"].round(1).astype(str) + "%)"
    tA2 = top3.groupby(["Region", "Period"])["txt"].apply("\n".join).unstack().fillna("")
    ref = "Table A.2" + cat_suffix(cat)
    add_table(tA2.reset_index(), ref,
              f"Dominant cluster transitions by oblast — {CAT_LABEL[cat]}",
              name="Table_A_2" if cat == CLUSTER_CATS[0] else f"Table_A_2_{cat}",
              float_fmt="{}",
              note="Up to three most frequent transitions per oblast and period; the percentage "
                   "is the share of that oblast's communities making the move.")

    frag = pdf.groupby(["Region", "Year"]).agg(active=("Cluster", "nunique"),
                                               n=("BUDGET_CODE", "nunique"))
    frag["per100"] = 100 * frag["active"] / frag["n"]
    RES[cat]["frag"] = frag

cats5 = [c for c in CLUSTER_CATS if c in RES]
if cats5:
    fig, axes = plt.subplots(2, len(cats5), figsize=(6 * len(cats5), 7), squeeze=False)
    for j, cat in enumerate(cats5):
        fr = RES[cat]["frag"]["per100"].unstack()
        for i, (mat, ttl, xl) in enumerate([
                (fr, "Fragmentation: active clusters per 100 communities", "Year"),
                (RES[cat]["reg"], "Instability: % of communities changing cluster", "Period")]):
            ax = axes[i, j]
            x = np.arange(mat.shape[1])
            q = mat.quantile([0.25, 0.5, 0.75])
            # the min-max band is deliberately not drawn: its width was set by a single
            # extreme oblast and it read as noise rather than as spread
            ax.fill_between(x, q.loc[0.25], q.loc[0.75], color=BLUE, alpha=0.28, lw=0,
                            label="middle half of oblasts (25th-75th percentile)")
            ax.plot(x, q.loc[0.5], color=BLUE, lw=2, marker="o", label="median oblast")
            ax.set_xticks(x, mat.columns, rotation=45 if i == 1 else 0)
            ax.set_title(f"{ttl}\n({CAT_LABEL[cat]})", fontsize=10)
            ax.set_xlabel(xl)
            style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    add_figure(fig, "Figure 5", "Regional dynamics of local budgets",
               name="fig5_regional_dynamics",
               note="Distribution across the 24 oblasts rather than 24 overlapping lines: the "
                    "line is the median oblast, the bands the interquartile and min–max ranges.")

# =============================================================================
# 10. FIGURE 4 — trajectory of one community
# =============================================================================

if "income" in RES:
    pdf, ch = RES["income"]["pdf"], RES["income"]["ch"]
    vol = ch.groupby("BUDGET_CODE")["Changed"].sum().sort_values(ascending=False)
    code = vol.index[0]
    log(f"\nmost volatile community: {name_of(code)} ({code}), {int(vol.iloc[0])} changes")
    traj = pdf[pdf["BUDGET_CODE"] == code].sort_values("Year")
    cent = pdf.groupby("Cluster")[["pc1", "pc2"]].mean()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(pdf["pc1"], pdf["pc2"], s=4, color=GREY, alpha=0.4, lw=0)
    for c, r in cent.iterrows():
        ax.text(r["pc1"], r["pc2"], str(int(c)), ha="center", va="center", fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec=DARK))
    ax.plot(traj["pc1"], traj["pc2"], color=RED, lw=2, marker="o", ms=6)
    for _, r in traj.iterrows():
        ax.annotate(str(int(r["Year"])), (r["pc1"], r["pc2"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    x1, x99 = np.percentile(pdf["pc1"], [1, 99])
    y1, y99 = np.percentile(pdf["pc2"], [1, 99])
    ax.set_xlim(min(x1, traj["pc1"].min()) - 1, max(x99, traj["pc1"].max()) + 1)
    ax.set_ylim(min(y1, traj["pc2"].min()) - 1, max(y99, traj["pc2"].max()) + 1)
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_title(f"Trajectory of {name_of(code)} (numbers mark cluster centroids)")
    style_axes(ax)
    plt.tight_layout()
    add_figure(fig, "Figure 4", f"Trajectory of {name_of(code)} in the component space",
               name="fig4_trajectory",
               note="Grey points are all community-years; numbered circles are cluster "
                    "centroids; the red line is the selected community year by year.")

# =============================================================================
# 11. FIGURE 6 — map of clusters in the last year (needs geopandas + geojson)
# =============================================================================

if "income" in RES:
    pdf = RES["income"]["pdf"]
    last = pdf[pdf["Year"] == pdf["Year"].max()]
    drawn = False
    if GEOJSON_L3 and os.path.exists(GEOJSON_L3):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(GEOJSON_L3)
            gdf[GEOJSON_KEY] = gdf[GEOJSON_KEY].astype(str).str.strip()
            m = gdf.merge(last[["BUDGET_CODE", "Cluster"]].astype({"BUDGET_CODE": str}),
                          left_on=GEOJSON_KEY, right_on="BUDGET_CODE", how="left")
            fig, ax = plt.subplots(figsize=(10, 7))
            m.plot(column="Cluster", categorical=True, legend=True, ax=ax,
                   cmap="tab10", missing_kwds={"color": "#eeeeee", "label": "no data"},
                   edgecolor="white", linewidth=0.15)
            ax.set_axis_off()
            ax.set_title(f"Revenue clusters, {int(last['Year'].max())}")
            add_figure(fig, "Figure 6", "Spatial distribution of revenue clusters",
                       name="fig6_map",
                       note="Drawn from the same clustering run as every table, so the cluster "
                            "numbers here match Table A.1 and Table A.2.")
            drawn = True
        except Exception as e:
            log(f"map skipped ({e})")
    if not drawn:
        comp = (pd.crosstab(last["Region"], last["Cluster"], normalize="index") * 100)
        fig, ax = plt.subplots(figsize=(1.0 + 0.55 * comp.shape[1], 0.32 * comp.shape[0] + 2))
        im = heatmap(ax, comp, fmt="{:.0f}", vmin=0, vmax=100)
        ax.set_xlabel("Cluster")
        ax.set_title(f"Share of each oblast's communities by cluster, "
                     f"{int(last['Year'].max())} (%)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="% of the oblast's communities")
        plt.tight_layout()
        add_figure(fig, "Figure 6", "Cluster composition by oblast in the last year",
                   name="fig6_cluster_composition",
                   note="Fallback for the map: set GEOJSON_L3 to viz_dashboard_data2/l3.geojson "
                        "and install geopandas to draw the choropleth instead. "
                        "clusters_income_last_year_for_map.csv feeds the dashboard map.")

# =============================================================================
# 12. FIGURE A.1 — how much variance the global components capture per region
# =============================================================================

if "income" in RES:
    ev_cats = [c for c in CLUSTER_CATS if c in RES]
    fig, axes = plt.subplots(1, len(ev_cats), figsize=(6 * len(ev_cats), 4.6),
                             squeeze=False)
    for j, cat in enumerate(ev_cats):
        shares = RES[cat]["model_shares"]
        Xs = StandardScaler().fit_transform(shares)
        n_hi = min(N_PC_HIGH, Xs.shape[1], Xs.shape[0])
        pca_hi = PCA(n_components=n_hi, random_state=SEED).fit(Xs)
        Z = pca_hi.transform(Xs)
        idx = shares.index.to_frame(index=False)
        idx["Region"] = idx["BUDGET_CODE"].astype(str).str[:2].map(REGION_MAP)

        rows = []
        for n_use in [N_PC, n_hi]:
            Xhat = Z[:, :n_use] @ pca_hi.components_[:n_use] + pca_hi.mean_
            resid = ((Xs - Xhat) ** 2).sum(axis=1)
            total = ((Xs - Xs.mean(axis=0)) ** 2).sum(axis=1)
            ev = pd.DataFrame({"Region": idx["Region"], "Year": idx["Year"],
                               "resid": resid, "total": total})
            g = ev.groupby(["Region", "Year"])[["resid", "total"]].sum()
            g["EV"] = 100 * (1 - g["resid"] / g["total"])
            for (r, y), v in g["EV"].items():
                rows.append({"Region": r, "Year": y, "n_components": n_use, "EV": v})
        ev_df = pd.DataFrame(rows)
        extra_csv(ev_df, f"explained_variance_by_region_{cat}.csv", index=False)

        ax = axes[0, j]
        # the two curves must be told apart at a glance: colour, dash AND marker differ
        for n_use, style, col, mk in [(N_PC, "-", BLUE, "o"), (n_hi, "--", RED, "s")]:
            sub = ev_df[ev_df["n_components"] == n_use].groupby("Year")["EV"]
            ax.plot(sub.median().index, sub.median().values, style, color=col, lw=2,
                    marker=mk, ms=5, label=f"median oblast, {n_use} components")
            ax.fill_between(sub.quantile(0.25).index, sub.quantile(0.25).values,
                            sub.quantile(0.75).values, color=col, alpha=0.13, lw=0)
        ax.set_title(f"Variance captured within oblasts — {CAT_LABEL[cat]}", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Explained variance, %")
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    plt.tight_layout()
    add_figure(fig, "Figure A.1", "Share of local variance captured by the global components",
               name="figA1_explained_variance",
               note="The components are estimated on the pooled 2018–2025 data, so within a "
                    "single oblast and year they capture only part of the variation. Bands are "
                    "the interquartile range across oblasts.")

# =============================================================================
# 13. TABLE A.3 — sensitivity to k
# =============================================================================

if not ARGS.quick:
    sens = {}
    for cat in RES:
        for k in K_SENS:
            p_k = pca_kmeans(RES[cat]["model_shares"], k=k,
                             weights=EXPERT_WEIGHTS.get(cat),
                             label_shares=RES[cat]["shares"])[0]
            col = f"{CAT_LABEL[cat].capitalize()}, k={k}"
            sens[col] = (transitions(p_k).groupby("Period")["Changed"].mean() * 100).round(1)
    if sens:
        add_table(pd.DataFrame(sens).reset_index(), "Table A.3",
                  "Sensitivity of the transition rate to the number of clusters",
                  name="Table_A_3",
                  note="Share of communities changing cluster (%) under k = "
                       + ", ".join(str(k) for k in K_SENS) +
                       ". Periods whose share moves little across k are robust findings; "
                       "periods that move a lot should not be over-interpreted.")

# =============================================================================
# 14. TABLE 2 + FIGURES 3, 7, 8, 9 — walk-forward forecasting
# =============================================================================

for cat in RES:
    pdf, pc_cols = RES[cat]["pdf"], RES[cat]["pc_cols"]
    p, num, dummies = ml_frame(pdf, pc_cols, K)
    feats = num + dummies
    log(f"\n================ WALK-FORWARD FORECAST: {cat.upper()} ================")

    bench, percls, shap_rows, first_model = [], [], [], None
    for T in FORECAST_YEARS:
        train = p[p["Year"] <= T - 2]
        test = p[p["Year"] == T - 1]
        if train.empty or test.empty or train["Target"].nunique() < 2:
            log(f"   {T}: skipped (not enough data)")
            continue
        y_tr, y_te = train["Target"].values, test["Target"].values
        changers = y_te != test["Cluster"].values
        unseen = ~np.isin(y_te, np.unique(y_tr))

        preds = {"Persistence (no change)": test["Cluster"].values}

        sc = StandardScaler().fit(train[num])
        Xtr = np.hstack([sc.transform(train[num]), train[dummies].values])
        Xte = np.hstack([sc.transform(test[num]), test[dummies].values])
        preds["Multinomial logit"] = LogisticRegression(max_iter=5000).fit(Xtr, y_tr).predict(Xte)

        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        preds["Random Forest"] = rf.fit(train[feats], y_tr).predict(test[feats])

        le = LabelEncoder().fit(y_tr)
        xg = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               random_state=SEED, tree_method="hist", n_jobs=-1)
        xg.fit(train[feats], le.transform(y_tr))
        preds["XGBoost"] = le.inverse_transform(xg.predict(test[feats]).astype(int))
        first_model = first_model or (xg, feats, T)

        for mname, yp in preds.items():
            bench.append({
                "Forecast year": T, "Model": mname,
                "Training observations": len(train), "Test communities": len(test),
                "Communities changing cluster (%)": round(100 * changers.mean(), 1),
                "Communities moving to a cluster unseen in training": int(unseen.sum()),
                "Accuracy": round(accuracy_score(y_te, yp), 3),
                "F1-macro": round(f1_score(y_te, yp, average="macro"), 3),
                "Accuracy on changers":
                    round(accuracy_score(y_te[changers], yp[changers]), 3) if changers.any() else np.nan,
            })

        # the same per-cluster metrics in-sample and out-of-sample: the gap between
        # the two rows of Figures 8 and 9 is what overfitting looks like
        yp_train = le.inverse_transform(xg.predict(train[feats]).astype(int))
        for split, y_true, y_pred in (("train", y_tr, yp_train),
                                      ("test", y_te, preds["XGBoost"])):
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            for c, mtr in rep.items():
                if str(c).isdigit():
                    percls.append({"Split": split, "Forecast year": T, "Cluster": int(c),
                                   "Recall": round(mtr["recall"], 2),
                                   "F1": round(mtr["f1-score"], 2),
                                   "Communities": int(mtr["support"])})

        if not ARGS.quick:
            sv = shap.TreeExplainer(xg).shap_values(test[feats])
            sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
            if sv.ndim == 2:
                sv = sv[:, :, None]
            imp = np.abs(sv).mean(axis=0).sum(axis=1)
            s = pd.Series(imp, index=feats).groupby(feature_group).sum()
            for f_, v in s.items():
                shap_rows.append({"Forecast year": T, "Feature": f_, "mean_abs_SHAP": v})

    bench = pd.DataFrame(bench)
    percls = pd.DataFrame(percls)
    ref = "Table 2" + cat_suffix(cat)
    add_table(bench, ref, f"Walk-forward forecasting benchmark — {CAT_LABEL[cat]}",
              name="Table_2" if cat == CLUSTER_CATS[0] else f"Table_2_{cat}",
              note="'Persistence' simply predicts that a community stays in its current cluster; "
                   "any model worth reporting has to beat it. 'Communities moving to a cluster "
                   "unseen in training' counts communities whose true next-year cluster never "
                   "appeared in the training years — no classifier can get those right, which is "
                   "the honest explanation of the accuracy in structural-break years.")

    # per-cluster scores and SHAP are kept here and drawn once, after the loop, so each
    # manuscript figure is a single file with one panel per category
    if not percls.empty:
        RES[cat]["percls"] = percls
        add_table(percls, "Table A.6" + cat_suffix(cat),
                  f"Per-cluster forecasting performance — {CAT_LABEL[cat]}",
                  name=f"Table_A_6_per_cluster_{cat}")

    if shap_rows:
        shp = pd.DataFrame(shap_rows)
        mat = shp.pivot(index="Feature", columns="Forecast year", values="mean_abs_SHAP").fillna(0)
        share = 100 * mat / mat.sum(axis=0)
        order = share.mean(axis=1).sort_values(ascending=False).index[:12]
        share = share.loc[order]
        RES[cat]["shap_share"] = share

        if "load" in RES[cat]:
            load = RES[cat]["load"]
            expl = []
            for T in share.columns:
                for f_ in share[T].sort_values(ascending=False).index[:3]:
                    base = f_.replace("prev_", "")
                    if base in load.columns:
                        top = load[base].reindex(
                            load[base].abs().sort_values(ascending=False).index).head(5)
                        expl.append({
                            "Forecast year": T, "Feature": f_,
                            "Share of importance, %": round(share.loc[f_, T], 1),
                            "Budget items behind it":
                                "; ".join(f"{c} {KEY_ITEMS.get(c, '')} ({w:+.2f})"
                                          for c, w in top.items())})
            if expl and cat == "income":
                add_table(pd.DataFrame(expl), "Table A.7",
                          "What the most important features actually contain",
                          name="Table_A_7_shap_explained",
                          note="Bridges Figure 7 and the budget data: for each year, the three "
                               "most important features and the items loading on them.")

    # ---- Figure 3: one tree of the fitted model
    if first_model and cat == "income":
        xg, feats, T = first_model
        drawn = False
        try:
            fig, ax = plt.subplots(figsize=(16, 9))
            xgb.plot_tree(xg, num_trees=0, ax=ax)
            ax.set_title(f"One decision tree of the XGBoost model (forecast year {T})")
            add_figure(fig, "Figure 3", "Internal structure of a forecasting tree",
                       name="fig3_tree", dpi=160,
                       note="Splits use the principal components and the cluster dummies; leaf "
                            "values are contributions to the score of the predicted cluster.")
            drawn = True
        except Exception as e:
            log(f"   xgb.plot_tree needs graphviz ({e}); saving the text dump instead")
        if not drawn:
            dump = xg.get_booster().get_dump()[0]
            with open(os.path.join(OUT_DIR, "fig3_tree.txt"), "w", encoding="utf-8") as fh:
                fh.write(dump)
            add_note("Figure 3", "Internal structure of a forecasting tree",
                     "graphviz is not installed, so the tree was written as text to "
                     "fig3_tree.txt. Install graphviz (pip install graphviz and the system "
                     "package) to get the picture.")

# ---- Figures 8 and 9: per-cluster recall and F1, one panel per category
pc_cats = [c for c in CLUSTER_CATS if "percls" in RES.get(c, {})]
for metric, ref_f, fname in [("Recall", "Figure 8", "fig8_recall_by_cluster"),
                             ("F1", "Figure 9", "fig9_f1_by_cluster")]:
    if not pc_cats:
        continue
    splits = ["train", "test"]
    mats = {}
    for cat in pc_cats:
        pc = RES[cat]["percls"]
        for sp in splits:
            mats[(sp, cat)] = pc[pc["Split"] == sp].pivot(
                index="Cluster", columns="Forecast year", values=metric)
    wide = sum(1.9 + 0.9 * mats[("test", c)].shape[1] for c in pc_cats)
    tall = 2 * max(0.40 * m.shape[0] + 1.7 for m in mats.values())
    fig, axes = plt.subplots(2, len(pc_cats), figsize=(wide, tall), squeeze=False,
                             layout="constrained")
    for i, sp in enumerate(splits):
        for j, cat in enumerate(pc_cats):
            mat = mats[(sp, cat)]
            im = heatmap(axes[i, j], mat, ylabels=[f"Cluster {c}" for c in mat.index])
            axes[i, j].set_title(
                f"{CAT_LABEL[cat].capitalize()} — "
                + ("TRAIN (years the model was fitted on)" if sp == "train"
                   else "TEST (the forecast year, never seen)"), fontsize=10)
            if i == 1:
                axes[i, j].set_xlabel("Forecast year")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.55, pad=0.015)
    fig.suptitle(f"XGBoost per-cluster {metric}: train vs test")
    add_figure(fig, ref_f, f"Per-cluster {metric}, train vs test", name=fname,
               note="Top row: the metric on the years the model was fitted on. Bottom row: the "
                    "forecast year the model never saw. A large gap between the rows is "
                    "overfitting; a blank cell means the cluster had no community that year. "
                    "These values and the aggregate in Table 2 come from the same run.")

# ---- Figure 7: SHAP importance, one panel per category
sh_cats = [c for c in CLUSTER_CATS if "shap_share" in RES.get(c, {})]
if sh_cats:
    wide = sum(2.2 + 0.9 * RES[c]["shap_share"].shape[1] for c in sh_cats)
    tall = max(0.42 * RES[c]["shap_share"].shape[0] + 2 for c in sh_cats)
    fig, axes = plt.subplots(1, len(sh_cats), figsize=(wide, tall), squeeze=False)
    for j, cat in enumerate(sh_cats):
        share = RES[cat]["shap_share"]
        im = heatmap(axes[0, j], share, fmt="{:.0f}", vmin=0,
                     vmax=float(share.values.max()))
        axes[0, j].set_xlabel("Forecast year")
        axes[0, j].set_title(f"{CAT_LABEL[cat].capitalize()}", fontsize=10)
        fig.colorbar(im, ax=axes[0, j], shrink=0.8)
    fig.suptitle("SHAP importance, % of the yearly total", y=0.99)
    plt.tight_layout()
    add_figure(fig, "Figure 7", "Feature importance (SHAP)", name="fig7_shap",
               note="The model's features are principal components, not budget items, so SHAP "
                    "ranks components. Table A.4 (in extra/) translates the leading components "
                    "into the budget items they are built from.")

# =============================================================================
# 15. RQ3 — is the 2024–2025 break about PIT or about transfers?
# =============================================================================

if "income" in RES:
    log("\n================ RQ3: military PIT vs transfers ================")
    shares, pdf = RES["income"]["shares"], RES["income"]["pdf"]
    sh = shares.reset_index()
    cols = [c for c in KEY_ITEMS if c in shares.columns]

    yearly = (sh.groupby("Year")[cols].mean() * 100).round(2)
    yearly.columns = [f"{c} — {KEY_ITEMS[c]}" for c in cols]
    add_table(yearly.T.reset_index().rename(columns={"index": "Budget item"}), "Table A.8",
              "Mean share of the key budget items in community revenues, by year (%)",
              name="Table_A_8_key_shares",
              note="The direct check behind RQ3: if the military PIT was the trigger, its share "
                   "should collapse exactly when the clusters move.")

    fig, ax = plt.subplots(figsize=(9, 5))
    for c in [c for c in ["11010100", "11010200", "41020100", "41036300", "14040100"]
              if c in cols]:
        ax.plot(yearly.index, (sh.groupby("Year")[c].mean() * 100).values,
                marker="o", lw=2, label=f"{KEY_ITEMS[c]}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean share of the community budget, %")
    ax.set_title("Key revenue items over time")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    plt.tight_layout()
    add_figure(fig, "Figure 10", "Key revenue items over time",
               name="fig10_key_items",
               note="Optional extra figure. It is the simplest possible evidence for the RQ3 "
                    "story and does not depend on any model.")

    ch = RES["income"]["ch"]
    for period in sorted(ch["Period"].unique())[-2:]:
        y1 = int(period[:4])
        movers = ch[ch["Period"] == period][["BUDGET_CODE", "Changed"]]
        m = sh.merge(movers, on="BUDGET_CODE")
        m = m[m["Year"].between(y1 - 1, y1 + 1)]
        g = (m.groupby(["Changed", "Year"])[cols].mean() * 100).round(2)
        g.columns = [KEY_ITEMS[c] for c in cols]
        extra_csv(g.T, f"rq3_movers_vs_stayers_{period}.csv")
        log_df(f"movers vs stayers, {period} (mean shares, %)", g.T)

    if not ARGS.quick:
        variants = {"baseline": None,
                    "without the 2025 education codes": EDU_2025_CODES,
                    "without the military PIT": MILITARY_PIT}
        vt = {}
        for vname, drop in variants.items():
            raw_v, model_v, _ = prep_shares(DATA["income"], drop)
            p_v = pca_kmeans(model_v, weights=EXPERT_WEIGHTS["income"],
                             label_shares=raw_v)[0]
            vt[vname] = (transitions(p_v).groupby("Period")["Changed"].mean() * 100).round(1)
        add_table(pd.DataFrame(vt).reset_index(), "Table A.9",
                  "Transition rate when individual budget codes are excluded (%)",
                  name="Table_A_9_variants",
                  note="If dropping the military PIT flattens the 2024–2025 wave, that item is "
                       "what drives the break; if dropping the 2025 education codes does, the "
                       "break is largely a reclassification.")

# =============================================================================
# 16. TABLES 3, 4, 5 — weights and scoring
# =============================================================================

inc_w = list(EXPERT_WEIGHTS["income"].items())
prog_w = list(EXPERT_WEIGHTS["prog"].items())
w_rows = []
for i in range(max(len(inc_w), len(prog_w))):
    a = inc_w[i] if i < len(inc_w) else ("", "")
    b = prog_w[i] if i < len(prog_w) else ("", "")
    w_rows.append([a[0], item_label(a[0], "income"), a[1],
                   b[0], item_label(b[0], "prog"), b[1]])
t3 = pd.DataFrame(w_rows, columns=["Revenue code", "Revenue item", "Weight",
                                   "Program code", "Program expenditure item", "Weight"])
add_table(t3, "Table 3", "Weight coefficients used in the municipal score",
          note="Positive weights mark own-source revenue and development spending, negative "
               "weights mark dependence on transfers and distress spending.")

for num, cat in enumerate([c for c in CLUSTER_CATS if c in RES], start=4):
    if cat not in EXPERT_WEIGHTS:
        log(f"scoring skipped for '{cat}': no expert weights")
        continue
    ref = f"Table {num}"
    raw_shares, pdf, pc_cols = (RES[cat]["shares"], RES[cat]["pdf"], RES[cat]["pc_cols"])
    w = EXPERT_WEIGHTS[cat]
    last = pdf[pdf["Year"] == pdf["Year"].max()]
    cent = last.groupby("Cluster")[pc_cols].mean()

    # The score is read straight off the actual budget shares: for every community,
    # sum(weight of the item x its share of the budget), then average within a cluster.
    # The earlier version went through the PCA loadings, which silently produced NaN
    # once the weighted codes stopped appearing among the leading loadings, and which
    # could disagree with the cluster numbering in section 4 — the numbering uses
    # exactly this quantity, so now the two can no longer contradict each other.
    cols = [c for c in raw_shares.columns if c in w]
    if not cols:
        log(f"   {ref}: none of the weighted codes for '{cat}' are present in the data "
            f"— scoring skipped")
        continue
    missing = [c for c in w if c not in raw_shares.columns]
    if missing:
        log(f"   {ref}: weighted codes absent from the data and therefore ignored: "
            + ", ".join(missing))
    sh = raw_shares.reset_index().merge(last[["BUDGET_CODE", "Year", "Cluster"]],
                                        on=["BUDGET_CODE", "Year"])
    sh["w_score"] = (sh[cols] * pd.Series({c: w[c] for c in cols})).sum(axis=1)
    per_cluster = sh.groupby("Cluster")["w_score"].mean()
    denom = per_cluster.abs().max()
    score = (100 * per_cluster / denom).round(1) if denom else per_cluster.round(1)

    head = "Cluster (revenues)" if cat == "income" else "Cluster (expenditures)"
    rows = []
    for c in score.sort_values(ascending=False).index:
        mem = last[last["Cluster"] == c]
        dist = np.linalg.norm(mem[pc_cols].values - cent.loc[c].values, axis=1)
        closest = mem.iloc[np.argsort(dist)[:3]]["BUDGET_CODE"].map(name_of).tolist()
        rows.append({head: f"Cluster {int(c)}", "Score": score[c],
                     "Three communities closest to the centroid": ", ".join(closest)})
        log(f"   {ref}: cluster {int(c)} — {len(mem)} communities, score {score[c]}")
    add_table(pd.DataFrame(rows), ref,
              f"Municipal ranking by {CAT_LABEL[cat]} structure, {int(last['Year'].max())}",
              note="Score = sum over budget items of (weight from Table 3 x the item's mean "
                   "share of the budget), averaged over the cluster's communities and rescaled "
                   "so the most extreme cluster is ±100. It is the same quantity that orders "
                   "the cluster numbers, so Table 3, the numbering and this table agree by "
                   "construction. Scores rank clusters, not individual communities.")

# =============================================================================
# 17. APPENDIX B — Poland (optional, runs only if the files are configured)
# =============================================================================

if POLAND_PIT and os.path.exists(POLAND_PIT):
    pit = pd.read_csv(POLAND_PIT)
    top = (pit.groupby("Powiat")["PIT"].max().sort_values(ascending=False).head(10).index)
    fig, ax = plt.subplots(figsize=(9, 5))
    for p_ in top:
        s = pit[pit["Powiat"] == p_].sort_values("Year")
        ax.plot(s["Year"], s["PIT"], marker="o", lw=1.8, label=p_)
    ax.set_xlabel("Year")
    ax.set_ylabel("PLN, million")
    ax.set_title("PIT revenues, ten largest powiats")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    style_axes(ax)
    plt.tight_layout()
    add_figure(fig, "Figure B.1", "PIT revenues of the largest Polish powiats",
               name="figB1_poland_pit")
else:
    add_note("Figure B.1", "PIT revenues of the largest Polish powiats",
             "skipped — set POLAND_PIT to a CSV with columns Year, Powiat, PIT")

if POLAND_PESEL and os.path.exists(POLAND_PESEL):
    pes = pd.read_csv(POLAND_PESEL)
    piv = pes.pivot_table(index="Powiat", columns="Year", values="Registered", aggfunc="sum")
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).head(15).index]
    fig, ax = plt.subplots(figsize=(9, 5))
    for y in piv.columns:
        ax.plot(range(len(piv)), piv[y].values, marker="o", lw=1.6, label=str(y))
    ax.set_xticks(range(len(piv)), piv.index, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Registered citizens of Ukraine")
    ax.set_title("PESEL UKR registrations, fifteen largest powiats")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    plt.tight_layout()
    add_figure(fig, "Figure B.2", "Registered Ukrainian citizens by powiat",
               name="figB2_poland_pesel")
else:
    add_note("Figure B.2", "Registered Ukrainian citizens by powiat",
             "skipped — set POLAND_PESEL to a CSV with columns Year, Powiat, Registered")

# =============================================================================
# 18. report.html + manifest
# =============================================================================

def sort_key(item):
    """Figures 1..n, Figures A, Figures B, then Tables 1..n and Tables A —
    the order the manuscript's 'Figure and Table Legends' uses."""
    m = re.match(r"^(Table|Figure)\s+(?:([AB])\.)?(\d+)([a-z]*)$", item["ref"])
    if not m:
        return (9, 9, 999, "")
    kind = 0 if m.group(1) == "Figure" else 1
    appendix = {None: 0, "A": 1, "B": 2}[m.group(2)]
    return (kind, appendix, int(m.group(3)), m.group(4))

REPORT.sort(key=sort_key)

manifest = pd.DataFrame([{"Reference": r["ref"], "Title": r["title"],
                          "File": r["path"], "Kind": r["kind"]} for r in REPORT])
manifest.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)

css = """
:root{--ink:#1b1b19;--muted:#6b6a66;--line:#e3e1dc;--bg:#faf9f7;--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:28px 20px;border-bottom:1px solid var(--line);background:#fff}
h1{margin:0 0 6px;font-size:24px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
main{max-width:1100px;margin:0 auto;padding:0 20px 60px}
nav{position:sticky;top:0;background:rgba(250,249,247,.95);backdrop-filter:blur(6px);
    border-bottom:1px solid var(--line);padding:10px 0;margin-bottom:24px;z-index:5}
nav a{display:inline-block;margin:2px 10px 2px 0;font-size:12px;color:var(--accent);
      text-decoration:none}
nav a:hover{text-decoration:underline}
section{background:#fff;border:1px solid var(--line);border-radius:10px;
        padding:20px;margin:18px 0}
h2{margin:0 0 4px;font-size:17px}
.ref{color:var(--accent);font-weight:600}
.note{color:var(--muted);font-size:13px;margin:8px 0 14px}
.file{font-size:12px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,monospace}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
.tablewrap{overflow-x:auto;max-height:520px;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;font-size:12.5px;width:100%}
th{position:sticky;top:0;background:#f2f1ee;text-align:left;padding:7px 10px;
   border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid #f0efeb;vertical-align:top}
tr:hover td{background:#fbfaf8}
.skip{color:#9a6b00;background:#fff8e6;border:1px solid #f0e0b0;padding:10px 12px;
      border-radius:6px;font-size:13px}
@media (prefers-color-scheme:dark){
  :root{--ink:#e8e6e1;--muted:#a3a19b;--line:#33322f;--bg:#1a1a18}
  header,section{background:#232220}
  nav{background:rgba(26,26,24,.95)}
  th{background:#2c2b28}
  tr:hover td{background:#2a2926}
  .skip{color:#e8c87a;background:#2e2a1e;border-color:#4a422c}
}
"""

parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local budget profiling — all results</title><style>{css}</style></head><body>
<header><h1>Local budget profiling under systemic shocks — generated results</h1>
<div class="sub">{datetime.datetime.now():%Y-%m-%d %H:%M} &middot;
N_PC={N_PC} &middot; k={K} &middot; seed={SEED} &middot;
{len([r for r in REPORT if r['kind']=='table'])} tables,
{len([r for r in REPORT if r['kind']=='figure'])} figures &middot;
full log in summary.txt</div></header><main><nav>"""]

for r in REPORT:
    anchor = re.sub(r"[^a-z0-9]+", "-", r["ref"].lower())
    parts.append(f'<a href="#{anchor}">{r["ref"]}</a>')
parts.append("</nav>")

for r in REPORT:
    anchor = re.sub(r"[^a-z0-9]+", "-", r["ref"].lower())
    parts.append(f'<section id="{anchor}"><h2><span class="ref">{r["ref"]}.</span> '
                 f'{r["title"]}</h2>')
    if r["kind"] == "note":
        parts.append(f'<div class="skip">{r["note"]}</div>')
    else:
        if r["note"]:
            parts.append(f'<div class="note">{r["note"]}</div>')
        if r["kind"] == "figure":
            parts.append(f'<img src="{r["path"]}" alt="{r["ref"]}">')
        else:
            parts.append(f'<div class="tablewrap">{r["html"]}</div>')
        parts.append(f'<div class="file">{r["path"]}</div>')
    parts.append("</section>")

parts.append("</main></body></html>")
with open(os.path.join(OUT_DIR, "report.html"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(parts))

# ---- final audit: every numbered item, its file and its size on disk ---------
log("\n================ OUTPUT AUDIT ================")
audit, empty = [], []
for r in REPORT:
    if r["kind"] == "note" or not r["path"]:
        audit.append({"Reference": r["ref"], "File": "(not produced)", "kB": 0})
        continue
    stem = os.path.join(OUT_DIR, r["path"])
    for f_ in ([stem] if r["kind"] == "figure"
               else [stem, stem[:-4] + ".xlsx"]):
        size = os.path.getsize(f_) if os.path.exists(f_) else -1
        audit.append({"Reference": r["ref"], "File": os.path.relpath(f_, OUT_DIR),
                      "kB": round(size / 1024, 1) if size >= 0 else "missing"})
        if size == 0:
            empty.append(os.path.relpath(f_, OUT_DIR))
log_df("every item and the size it actually has on disk", pd.DataFrame(audit), max_rows=200)
if empty:
    log("WARNING these files are 0 bytes: " + ", ".join(empty)
        + "  -> the output volume is full or not writable; free space and re-run.")
else:
    log("OK: every file written has a non-zero size")

missing = sorted(MANUSCRIPT_REFS - {r["ref"] for r in REPORT})
if missing:
    log("not produced in this run: " + ", ".join(missing)
        + ("  (--quick skips Figure 7 and Table A.3)" if ARGS.quick else "")
        + "  Figure 6 falls back to a heatmap without geopandas, Figure 3 to a text dump "
          "without graphviz, and Figures B.1/B.2 need POLAND_PIT / POLAND_PESEL.")
stray = sorted(f for f in os.listdir(OUT_DIR)
               if os.path.isfile(os.path.join(OUT_DIR, f))
               and not re.match(r"^(Table_|fig)", f)
               and f not in ("report.html", "manifest.csv", "summary.txt")
               and not f.startswith("cache_"))
if stray:
    log("note: unexpected files in the output folder: " + ", ".join(stray))

log("\n" + "=" * 70)
log(f"done — {len(manifest)} items written to {OUT_DIR}/")
log("open:", os.path.abspath(os.path.join(OUT_DIR, "report.html")))
log("=" * 70)
log_df("manifest", manifest)
LOG.close()