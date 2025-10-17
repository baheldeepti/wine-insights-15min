"""
Wine Insights Dashboard (Streamlit)

This Streamlit application loads a wine dataset and provides a quick dashboard
with interactive filters and simple charts. It is designed for a 15-minute
demo for beginners learning how to build dashboards with Streamlit.

Key features:
* Load the dataset from a local path, GitHub, KaggleHub, or a mounted volume
  using UTF-16 encoding (the dataset includes non-ASCII characters).
* Automatically map column names from the dataset to logical fields (price,
  rating, color, location, etc.).
* Parse the ``Countrys`` field to extract grape variety and location
  information when no dedicated columns exist.
* Display filters (Region, Variety, Color, Year) across the top of the
  dashboard using drop-down multi-selects and a slider.
* Show basic KPIs: total wines, average rating, median price, and average
  ABV or count of locations.
* Plot simple charts using Matplotlib: Top Locations, Average Rating by
  Year, Price Distribution, and Top Varieties.

To run this app:
    streamlit run app.py

Ensure that the CSV file ``vivno_dataset.csv`` is placed either in a
``data/`` directory relative to this script or available in the remote
repository specified below.  See the ``load_data`` function for details.
"""

import os
import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# 0) Streamlit Page Settings
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Wine Insights — quick dashboard", layout="wide")


# ----------------------------------------------------------------------------
# 1) Data Loading (UTF-16)
# ----------------------------------------------------------------------------
def try_kagglehub_download() -> str | None:
    """Attempt to download the dataset from KaggleHub.

    Returns
    -------
    str | None
        Absolute path to the downloaded CSV, or ``None`` if download fails.
    """
    try:
        import kagglehub
        base_path = kagglehub.dataset_download(
            "salohiddindev/wine-dataset-scraping-from-wine-com"
        )
        candidate_names = [
            "vivno_dataset.csv",
            "vivno_data.csv",
            "vivino_dataset.csv",
            "vivno_dataset (1).csv",
        ]
        for name in candidate_names:
            candidate = pathlib.Path(base_path) / name
            if candidate.exists():
                return str(candidate)
        for csv_file in pathlib.Path(base_path).glob("*.csv"):
            return str(csv_file)
    except Exception:
        return None
    return None


def load_data() -> tuple[pd.DataFrame | None, str]:
    """Load the wine dataset, trying local, GitHub, KaggleHub, and /mnt/data."""
    # 1) Local repo path (data folder)
    local_path = pathlib.Path("data") / "vivno_dataset.csv"
    if local_path.exists():
        try:
            df_local = pd.read_csv(local_path, encoding="utf-16", on_bad_lines="skip")
            return df_local, "✅ Loaded local data/vivno_dataset.csv"
        except Exception as e:
            return None, f"❌ Error reading local dataset: {e}"

    # 2) GitHub raw URL
    github_url = (
        "https://raw.githubusercontent.com/baheldeepti/wine-insights-15min/main/vivno_dataset.csv"
    )
    try:
        df_github = pd.read_csv(github_url, encoding="utf-16", on_bad_lines="skip")
        return df_github, "✅ Loaded data from GitHub repository"
    except Exception:
        pass

    # 3) KaggleHub fallback
    kaggle_path = try_kagglehub_download()
    if kaggle_path is not None and os.path.exists(kaggle_path):
        try:
            df_kaggle = pd.read_csv(kaggle_path, encoding="utf-16", on_bad_lines="skip")
            return df_kaggle, f"✅ Loaded data via KaggleHub: {kaggle_path}"
        except Exception as e:
            return None, f"❌ Error reading Kaggle dataset: {e}"

    # 4) Teaching environment mount
    mounted_path = "/mnt/data/vivno_dataset.csv"
    if os.path.exists(mounted_path):
        try:
            df_mounted = pd.read_csv(mounted_path, encoding="utf-16", on_bad_lines="skip")
            return df_mounted, "✅ Loaded /mnt/data/vivno_dataset.csv"
        except Exception as e:
            return None, f"❌ Error reading mounted dataset: {e}"

    return None, "❌ No CSV found (local/GitHub/Kaggle/mounted)."


df, load_message = load_data()
st.title("🍷 Wine Insights — quick dashboard")
st.caption(load_message)

if df is None or df.empty:
    st.error("Could not load any data. Please ensure the dataset is available.")
    st.stop()


# ----------------------------------------------------------------------------
# 2) Column mapping & parsing
# ----------------------------------------------------------------------------
def map_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, ".": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


price_col = map_column(df, ["Prices", "price", "wine_price", "price_usd"])
rating_col = map_column(df, ["Ratings", "rating", "points", "average_rating"])
ratings_ncol = map_column(df, ["Ratingsnum", "ratings_count", "n_ratings"])
country_col = map_column(df, ["Countrys", "country", "country_name", "region"])
color_col = map_column(df, ["color_wine", "color", "wine_color"])
abv_col = map_column(df, ["ABV %", "abv", "alcohol", "alcohol_percent"])
name_col = map_column(df, ["Names", "name", "title"])
variety_col = map_column(df, ["Variety", "variety", "grape", "grapes", "wine_variety"])

# Numeric coercions
if price_col:
    df[price_col] = coerce_numeric(df[price_col])
if rating_col:
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
if ratings_ncol:
    df[ratings_ncol] = pd.to_numeric(df[ratings_ncol], errors="coerce")
if abv_col:
    df[abv_col] = pd.to_numeric(df[abv_col], errors="coerce")

# Parse Countrys if present
if country_col:
    df["__location__"] = df[country_col].astype(str).str.extract(r"from\s+(.*)", expand=False)
    df["__location__"] = df["__location__"].fillna(df[country_col].astype(str))
    df["__location_last__"] = (
        df["__location__"].astype(str).str.split(",").str[-1].str.strip()
    )
    df["__variety__"] = df[country_col].astype(str).str.extract(
        r"^(.*?)\s+from\s", expand=False
    )
else:
    df["__location_last__"] = np.nan
    df["__variety__"] = np.nan

# Variety field preference
variety_field: str | None = variety_col if variety_col else "__variety__"

# Year from Names/title
if name_col:
    df["__year__"] = pd.to_numeric(
        df[name_col].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce"
    )
else:
    df["__year__"] = np.nan


# ----------------------------------------------------------------------------
# 3) Filters
# ----------------------------------------------------------------------------
st.markdown("### Filters")

# Region options
if "__location_last__" in df.columns:
    region_options = (
        df["__location_last__"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    region_options = sorted(set(region_options))
else:
    region_options = []

# Variety options
if variety_field and ((variety_field in df.columns) or variety_field == "__variety__"):
    variety_series = df[variety_field]
    variety_options = (
        variety_series.dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
    )
    variety_options = sorted(set(variety_options))
else:
    variety_options = []

# Color options
if color_col:
    color_options = (
        df[color_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    color_options = sorted(set(color_options))
else:
    color_options = []

# Year slider bounds
has_year = df["__year__"].notna().any()
if has_year:
    year_min = int(df["__year__"].min(skipna=True))
    year_max = int(df["__year__"].max(skipna=True))
else:
    year_min, year_max = 1900, 2025

col_f1, col_f2, col_f3, col_f4 = st.columns([1.2, 1.2, 0.9, 0.9])

with col_f1:
    selected_regions = st.multiselect("Region / State", options=region_options, default=[])

with col_f2:
    selected_varieties = st.multiselect("Variety / Grape", options=variety_options, default=[])

with col_f3:
    if color_options:
        selected_colors = st.multiselect("Color", options=color_options, default=[])
    else:
        selected_colors = []

with col_f4:
    if has_year:
        selected_year_range = st.slider(
            "Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max)
        )
    else:
        selected_year_range = None

# Apply filters
filter_mask = pd.Series(True, index=df.index)

if selected_regions and "__location_last__" in df.columns:
    filter_mask &= df["__location_last__"].astype(str).isin(selected_regions)

if selected_varieties and variety_field:
    if variety_field in df.columns:
        filter_mask &= df[variety_field].astype(str).isin(selected_varieties)
    else:
        filter_mask &= df["__variety__"].astype(str).isin(selected_varieties)

if selected_colors and color_col:
    filter_mask &= df[color_col].astype(str).isin(selected_colors)

if selected_year_range and has_year:
    min_year, max_year = selected_year_range
    filter_mask &= df["__year__"].between(min_year, max_year, inclusive="both")

df_filtered = df.loc[filter_mask].copy()


# ----------------------------------------------------------------------------
# 4) KPIs
# ----------------------------------------------------------------------------
st.markdown("### Key KPIs")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("Total Wines", f"{len(df_filtered):,}")

with kpi_col2:
    if rating_col and df_filtered[rating_col].notna().any():
        st.metric("Avg Rating", f"{df_filtered[rating_col].mean():.2f}")
    else:
        st.metric("Avg Rating", "—")

with kpi_col3:
    if price_col and df_filtered[price_col].notna().any():
        st.metric("Median Price", f"${df_filtered[price_col].median():,.2f}")
    else:
        st.metric("Median Price", "—")

with kpi_col4:
    if abv_col and df_filtered[abv_col].notna().any():
        st.metric("Avg ABV", f"{df_filtered[abv_col].mean():.1f}%")
    elif "__location_last__" in df_filtered.columns:
        st.metric("Locations", df_filtered["__location_last__"].nunique())
    else:
        st.metric("Locations", "—")

st.divider()


# ----------------------------------------------------------------------------
# 5) Charts (Matplotlib)
# ----------------------------------------------------------------------------

# A) Top Locations by Number of Wines
st.subheader("Top Locations by Number of Wines")
if "__location_last__" in df_filtered.columns:
    top_locations = (
        df_filtered["__location_last__"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )
    if not top_locations.empty:
        fig_loc = plt.figure(figsize=(8, 4.5))
        ax = top_locations.plot(kind="barh")  # <-- define ax
        plt.xlabel("Count")
        plt.ylabel("Location")
        # annotate bars (values on bars)
        for i, v in enumerate(top_locations.values):
            ax.text(v + (max(top_locations.values) * 0.01), i, str(v), va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_loc)
    else:
        st.info("No location data after filters.")
else:
    st.info("Location field not available in dataset.")

# B) Average Rating by Year
st.subheader("Average Rating by Year")
if rating_col and df_filtered["__year__"].notna().any():
    rating_by_year = (
        df_filtered.dropna(subset=["__year__"])[["__year__", rating_col]]
        .groupby("__year__")
        .mean()
    )
    if not rating_by_year.empty:
        fig_year = plt.figure(figsize=(8, 3.8))
        plt.plot(rating_by_year.index, rating_by_year[rating_col], marker="o")
        # annotate each point
        for x, y in zip(rating_by_year.index, rating_by_year[rating_col]):
            if pd.notna(y):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.tight_layout()
        st.pyplot(fig_year)
    else:
        st.info("No year/rating data after filters.")
else:
    st.info("Year or rating not available to plot the trend.")

# C) Price Distribution
st.subheader("Price Distribution (USD)")
if price_col and df_filtered[price_col].notna().any():
    fig_price = plt.figure(figsize=(8, 3.8))
    counts, bins, patches = plt.hist(df_filtered[price_col].dropna(), bins=30)
    # annotate bin tops with counts
    for count, left, right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            x = (left + right) / 2
            plt.text(x, count, str(int(count)), ha='center', va='bottom', fontsize=8)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig_price)
else:
    st.info("Price data not available to plot a distribution.")

# D) Top Varieties
st.subheader("Top Varieties")
if variety_field and variety_field in df_filtered.columns:
    series_var = df_filtered[variety_field]
else:
    series_var = df_filtered.get("__variety__", pd.Series([], dtype="object"))

if not series_var.empty:
    top_varieties = (
        series_var
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )
    if not top_varieties.empty:
        fig_var = plt.figure(figsize=(8, 4.5))
        ax = top_varieties.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Variety")
        for i, v in enumerate(top_varieties.values):
            ax.text(v + (max(top_varieties.values) * 0.01), i, str(v), va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_var)
    else:
        st.info("No variety data after filters.")
else:
    st.info("Variety field not available.")

st.divider()
st.caption(
    "Built for a 15-minute demo: explore the data with filters above, "
    "view summary KPIs, and see simple charts for locations, ratings, prices, "
    "and varieties."
)
