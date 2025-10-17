# app.py — Wine Insights (top filters demo)
# Python 3.11

import os
import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# 0) Streamlit Page Settings
# ----------------------------
st.set_page_config(page_title="Wine Insights — quick dashboard", layout="wide")

# ----------------------------
# 1) Data Loading (UTF-16)
# ----------------------------
def try_kagglehub_download():
    """Fallback: fetches dataset from Kaggle if local / GitHub not found."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("salohiddindev/wine-dataset-scraping-from-wine-com")
        # Try likely names; else pick any CSV in the folder
        for name in [
            "vivno_dataset.csv", "vivno_data.csv", "vivino_dataset.csv", "vivno_dataset (1).csv"
        ]:
            p = pathlib.Path(path) / name
            if p.exists():
                return str(p)
        any_csvs = list(pathlib.Path(path).glob("*.csv"))
        return str(any_csvs[0]) if any_csvs else None
    except Exception:
        return None

def load_data():
    """Load CSV with graceful fallbacks. This dataset is UTF-16."""
    # 1) Local repo path
    local_path = "data/vivno_dataset.csv"
    if os.path.exists(local_path):
        return pd.read_csv(local_path, encoding="utf-16", on_bad_lines="skip"), "✅ Loaded local data/vivno_dataset.csv"

    # 2) GitHub raw (repo root, as in your project)
    gh_url = "https://raw.githubusercontent.com/baheldeepti/wine-insights-15min/main/vivno_dataset.csv"
    try:
        df = pd.read_csv(gh_url, encoding="utf-16", on_bad_lines="skip")
        return df, "✅ Loaded data from GitHub repository"
    except Exception:
        pass

    # 3) KaggleHub
    kb = try_kagglehub_download()
    if kb and os.path.exists(kb):
        return pd.read_csv(kb, encoding="utf-16", on_bad_lines="skip"), f"Loaded via KaggleHub: {kb}"

    # 4) Teaching env mount
    mounted = "/mnt/data/vivno_dataset.csv"
    if os.path.exists(mounted):
        return pd.read_csv(mounted, encoding="utf-16", on_bad_lines="skip"), "Loaded /mnt/data/vivno_dataset.csv"

    return None, "❌ No CSV found (local/GitHub/Kaggle)."

df, load_msg = load_data()

st.title("🍷 Wine Insights — quick dashboard")
st.caption(load_msg)

if df is None or df.empty:
    st.error("Could not load any data. Please put your CSV at **data/vivno_dataset.csv**.")
    st.stop()

# ----------------------------
# 2) Column mapping & parsing
# ----------------------------
def map_column(df, options):
    lower = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower:
            return lower[opt.lower()]
    return None

def coerce_numeric(s):
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
         .str.replace(r"[^0-9.\-]", "", regex=True)
         .replace({"": np.nan, ".": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")

# Map dataset-specific names
price_col    = map_column(df, ["Prices", "price", "wine_price", "price_usd"])
rating_col   = map_column(df, ["Ratings", "rating", "points", "average_rating"])
ratings_ncol = map_column(df, ["Ratingsnum", "ratings_count", "n_ratings"])
country_col  = map_column(df, ["Countrys", "country", "country_name", "region"])
color_col    = map_column(df, ["color_wine", "color", "wine_color"])
abv_col      = map_column(df, ["ABV %", "abv", "alcohol", "alcohol_percent"])
name_col     = map_column(df, ["Names", "name", "title"])
variety_col  = map_column(df, ["Variety", "variety", "grape", "grapes", "wine_variety"])

# Coerce numeric fields
if price_col:    df[price_col] = coerce_numeric(df[price_col])
if rating_col:   df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
if ratings_ncol: df[ratings_ncol] = pd.to_numeric(df[ratings_ncol], errors="coerce")
if abv_col:      df[abv_col] = pd.to_numeric(df[abv_col], errors="coerce")

# Parse short location and variety from "Countrys" like "Chardonnay from Sonoma County, California"
if country_col:
    # part after "from "
    df["__location__"] = df[country_col].astype(str).str.extract(r"from\s+(.*)", expand=False)
    df["__location__"] = df["__location__"].fillna(df[country_col].astype(str))
    # last token after comma (e.g., "California")
    df["__location_last__"] = df["__location__"].astype(str).str.split(",").str[-1].str.strip()
    # simple variety from text before " from "
    df["__variety__"] = df[country_col].astype(str).str.extract(r"^(.*?)\s+from\s", expand=False)
else:
    df["__location_last__"] = np.nan
    df["__variety__"] = np.nan

# Prefer a real variety column; else use parsed one
variety_field = variety_col if variety_col else "__variety__"

# Year from Names/title (if present)
if name_col:
    df["__year__"] = pd.to_numeric(
        df[name_col].astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
else:
    df["__year__"] = np.nan

# ----------------------------
# 3) Filters — ON TOP
# ----------------------------
st.markdown("### Filters")

# Build the lists safely
regions = (
    df["__location_last__"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
    if "__location_last__" in df.columns else []
)
regions = sorted(set(regions))

varieties = (
    df[variety_field].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
    if variety_field in df.columns or variety_field == "__variety__" else []
)
varieties = sorted(set(varieties))

year_has_any = df["__year__"].notna().any()
if year_has_any:
    y_min = int(df["__year__"].min(skipna=True))
    y_max = int(df["__year__"].max(skipna=True))
else:
    y_min, y_max = 1900, 2025  # benign defaults

# Top-row filter UI
c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.9, 0.9])
with c1:
    sel_regions = st.multiselect(
        "Region / State",
        options=regions,
        default=regions[: min(8, len(regions))] if regions else []
    )
with c2:
    sel_varieties = st.multiselect(
        "Variety / Grape",
        options=varieties,
        default=varieties[: min(8, len(varieties))] if varieties else []
    )
with c3:
    if color_col:
        colors = (
            df[color_col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
        )
        colors = sorted(set(colors))
        sel_colors = st.multiselect("Color", options=colors, default=colors[: min(3, len(colors))])
    else:
        sel_colors = []
with c4:
    if year_has_any:
        yr = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))
    else:
        yr = None

# Apply filters
filt = pd.Series(True, index=df.index)

if sel_regions and "__location_last__" in df.columns:
    filt &= df["__location_last__"].astype(str).isin(sel_regions)

if sel_varieties and (variety_field in df.columns or variety_field == "__variety__"):
    filt &= df[variety_field].astype(str).isin(sel_varieties)

if sel_colors and color_col:
    filt &= df[color_col].astype(str).isin(sel_colors)

if yr and df["__year__"].notna().any():
    filt &= df["__year__"].between(yr[0], yr[1], inclusive="both")

df_f = df.loc[filt].copy()

# ----------------------------
# 4) KPIs
# ----------------------------
st.markdown("### Key KPIs")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Wines", f"{len(df_f):,}")
with k2:
    if rating_col and df_f[rating_col].notna().any():
        st.metric("Avg Rating", f"{df_f[rating_col].mean():.2f}")
    else:
        st.metric("Avg Rating", "—")
with k3:
    if price_col and df_f[price_col].notna().any():
        st.metric("Median Price", f"${df_f[price_col].median():,.2f}")
    else:
        st.metric("Median Price", "—")
with k4:
    if abv_col and df_f[abv_col].notna().any():
        st.metric("Avg ABV", f"{df_f[abv_col].mean():.1f}%")
    elif "__location_last__" in df_f.columns:
        st.metric("Locations", df_f["__location_last__"].nunique())
    else:
        st.metric("Locations", "—")

st.divider()

# ----------------------------
# 5) Charts (matplotlib)
# ----------------------------

# A) Top Locations
st.subheader("Top Locations by Number of Wines")
if "__location_last__" in df_f.columns:
    top_loc = (
        df_f["__location_last__"].dropna().astype(str).str.strip().replace("", np.nan).dropna()
        .value_counts().head(10).sort_values(ascending=True)
    )
    if not top_loc.empty:
        fig = plt.figure(figsize=(8, 4.5))
        top_loc.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Location")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No location data after filters.")
else:
    st.info("Location field not available in dataset.")

# B) Average Rating by Year
st.subheader("Average Rating by Year")
if rating_col and df_f["__year__"].notna().any():
    grp = df_f.dropna(subset=["__year__"])[["__year__", rating_col]].groupby("__year__").mean()
    if not grp.empty:
        fig2 = plt.figure(figsize=(8, 3.8))
        plt.plot(grp.index, grp[rating_col], marker="o")
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("No year/rating data after filters.")
else:
    st.info("Year or rating not available to plot the trend.")

# C) Price Distribution
st.subheader("Price Distribution (USD)")
if price_col and df_f[price_col].notna().any():
    fig3 = plt.figure(figsize=(8, 3.8))
    plt.hist(df_f[price_col].dropna(), bins=30)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig3)
else:
    st.info("Price data not available to plot a distribution.")

# D) Top Varieties
st.subheader("Top Varieties")
if variety_field and variety_field in df_f.columns:
    series = df_f[variety_field]
else:
    series = df_f.get("__variety__", pd.Series([], dtype="object"))

if not series.empty:
    top_var = (
        series.dropna().astype(str).str.strip().replace("", np.nan).dropna()
        .value_counts().head(10).sort_values(ascending=True)
    )
    if not top_var.empty:
        fig4 = plt.figure(figsize=(8, 4.5))
        top_var.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Variety")
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        st.info("No variety data after filters.")
else:
    st.info("Variety field not available.")

st.divider()
st.caption("Built for a 15-minute demo: top filters, KPIs, and three classic charts. Change filters above to explore.")
