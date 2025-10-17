"""
Wine Insights Dashboard (Streamlit)

This Streamlit application loads a wine dataset and provides a quick dashboard
with interactive filters and simple charts. It is designed for a 15‑minute
demo for beginners learning how to build dashboards with Streamlit.

Key features:
* Load the dataset from a local path, GitHub, KaggleHub, or a mounted volume
  using UTF‑16 encoding (the dataset includes non‑ASCII characters).
* Automatically map column names from the dataset to logical fields (price,
  rating, color, location, etc.).
* Parse the ``Countrys`` field to extract grape variety and location
  information when no dedicated columns exist.
* Display filters (Region, Variety, Color, Year) across the top of the
  dashboard using drop‑down multi‑selects and a slider.
* Show basic KPIs: total wines, average rating, median price, and average
  ABV or count of locations.
* Plot simple charts using Matplotlib: Top Locations, Average Rating by
  Year, Price Distribution, and Top Varieties.

To run this app:

```
streamlit run app.py
```

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
# 1) Data Loading (UTF‑16)
#
# The dataset is encoded in UTF‑16.  This helper attempts to load the CSV
# from several locations: a local ``data`` folder, a GitHub raw URL, Kaggle,
# or a mounted volume in teaching environments.  The first location that
# succeeds will be used.
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
        # Attempt to download the dataset using kagglehub.  The path
        # returned by kagglehub.dataset_download is a directory containing
        # the dataset files.  We search for plausible CSV names.
        base_path = kagglehub.dataset_download(
            "salohiddindev/wine-dataset-scraping-from-wine-com"
        )
        # Try common filenames first, then any CSV in the directory
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
        # Fallback: return the first CSV found
        for csv_file in pathlib.Path(base_path).glob("*.csv"):
            return str(csv_file)
    except Exception:
        # Import or download failure: return None
        return None
    return None


def load_data() -> tuple[pd.DataFrame | None, str]:
    """Load the wine dataset.

    This function checks several locations in order:
    1. ``data/vivno_dataset.csv`` relative to this script.
    2. A GitHub raw URL pointing to the user's repository.
    3. KaggleHub (if available).
    4. A mounted path at ``/mnt/data/vivno_dataset.csv``.

    Returns
    -------
    tuple[pd.DataFrame | None, str]
        A tuple containing the DataFrame (or ``None`` on failure) and an
        informative status message.
    """
    # 1) Local repo path (data folder)
    local_path = pathlib.Path("data") / "vivno_dataset.csv"
    if local_path.exists():
        try:
            df_local = pd.read_csv(local_path, encoding="utf-16", on_bad_lines="skip")
            return df_local, "✅ Loaded local data/vivno_dataset.csv"
        except Exception as e:
            return None, f"❌ Error reading local dataset: {e}"

    # 2) GitHub raw URL (points to repo root)
    github_url = (
        "https://raw.githubusercontent.com/baheldeepti/wine-insights-15min/main/vivno_dataset.csv"
    )
    try:
        df_github = pd.read_csv(github_url, encoding="utf-16", on_bad_lines="skip")
        return df_github, "✅ Loaded data from GitHub repository"
    except Exception:
        # Continue to next fallback
        pass

    # 3) KaggleHub fallback
    kaggle_path = try_kagglehub_download()
    if kaggle_path is not None and os.path.exists(kaggle_path):
        try:
            df_kaggle = pd.read_csv(kaggle_path, encoding="utf-16", on_bad_lines="skip")
            return df_kaggle, f"✅ Loaded data via KaggleHub: {kaggle_path}"
        except Exception as e:
            return None, f"❌ Error reading Kaggle dataset: {e}"

    # 4) Teaching environment mount (as used in OpenAI Code Interpreter)
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
#
# The dataset is scraped, so column names may differ.  We map expected
# logical fields (price, rating, etc.) to the actual column names in the
# DataFrame by checking several possible names.  We also parse useful
# information from the ``Countrys`` field when dedicated columns are absent.
# ----------------------------------------------------------------------------
def map_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from a list of candidates.

    Parameters
    ----------
    frame : pandas.DataFrame
        The DataFrame in which to search.
    candidates : list[str]
        A list of potential column names (case‑insensitive).

    Returns
    -------
    str | None
        The matched column name from ``frame`` if found; otherwise ``None``.
    """
    lower_map = {c.lower(): c for c in frame.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values, stripping non‑numeric characters.

    Strings like ``'$12.99'``, ``'95 points'`` and ``'12,99'`` are cleaned
    and converted to floats.  Empty strings and isolated punctuation are
    replaced with NaN.

    Parameters
    ----------
    series : pandas.Series
        The series to convert.

    Returns
    -------
    pandas.Series
        The coerced numeric series.
    """
    # If already numeric dtype, just coerce to numeric (no cleaning needed)
    if series.dtype.kind in "biufc":
        return pd.to_numeric(series, errors="coerce")
    # Otherwise, remove anything that isn't a digit, period, or minus sign
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, ".": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


# Map dataset‑specific columns to logical names
price_col = map_column(df, ["Prices", "price", "wine_price", "price_usd"])
rating_col = map_column(df, ["Ratings", "rating", "points", "average_rating"])
ratings_ncol = map_column(df, ["Ratingsnum", "ratings_count", "n_ratings"])
country_col = map_column(df, ["Countrys", "country", "country_name", "region"])
color_col = map_column(df, ["color_wine", "color", "wine_color"])
abv_col = map_column(df, ["ABV %", "abv", "alcohol", "alcohol_percent"])
name_col = map_column(df, ["Names", "name", "title"])
variety_col = map_column(df, ["Variety", "variety", "grape", "grapes", "wine_variety"])

# Convert numeric fields where necessary
if price_col:
    df[price_col] = coerce_numeric(df[price_col])
if rating_col:
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
if ratings_ncol:
    df[ratings_ncol] = pd.to_numeric(df[ratings_ncol], errors="coerce")
if abv_col:
    df[abv_col] = pd.to_numeric(df[abv_col], errors="coerce")


# Parse the ``Countrys`` column when present
if country_col:
    # Extract the portion after 'from ' (e.g., 'Sonoma County, California')
    df["__location__"] = df[country_col].astype(str).str.extract(r"from\s+(.*)", expand=False)
    # If the regex didn't find anything (no 'from '), fall back to the entire string
    df["__location__"] = df["__location__"].fillna(df[country_col].astype(str))
    # ``__location_last__``: take the last token after the final comma (e.g., 'California')
    df["__location_last__"] = (
        df["__location__"].astype(str).str.split(",").str[-1].str.strip()
    )
    # ``__variety__``: text before ' from ' (the grape or blend)
    df["__variety__"] = df[country_col].astype(str).str.extract(
        r"^(.*?)\s+from\s", expand=False
    )
else:
    # If ``Countrys`` column is not present, create empty columns for consistency
    df["__location_last__"] = np.nan
    df["__variety__"] = np.nan

# Determine which column to use for grape variety: a real column or parsed one
variety_field: str | None
if variety_col:
    variety_field = variety_col
else:
    variety_field = "__variety__"

# Extract four‑digit year from the ``name_col`` (Names) if available
if name_col:
    df["__year__"] = pd.to_numeric(
        df[name_col].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce"
    )
else:
    df["__year__"] = np.nan


# ----------------------------------------------------------------------------
# 3) Filters (Positioned on the top row)
#
# Build the filter lists from the dataset.  We leave the default selection
# empty so that all rows are shown initially.  This avoids the charts
# appearing empty due to pre‑selected filters.
# ----------------------------------------------------------------------------
st.markdown("### Filters")

# Regions/states: use the last token from the location string
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

# Varieties/grapes: use the chosen variety field (real column or parsed)
if variety_field and (
    (variety_field in df.columns) or variety_field == "__variety__"
):
    variety_series = df[variety_field]
    variety_options = (
        variety_series
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    variety_options = sorted(set(variety_options))
else:
    variety_options = []

# Colors (wine category) if available
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

# Year slider boundaries
has_year = df["__year__"].notna().any()
if has_year:
    year_min = int(df["__year__"].min(skipna=True))
    year_max = int(df["__year__"].max(skipna=True))
else:
    # Provide reasonable defaults if no years are available
    year_min, year_max = 1900, 2025

# Display filters across the top using columns
col_f1, col_f2, col_f3, col_f4 = st.columns([1.2, 1.2, 0.9, 0.9])

with col_f1:
    selected_regions = st.multiselect(
        "Region / State", options=region_options, default=[]
    )

with col_f2:
    selected_varieties = st.multiselect(
        "Variety / Grape", options=variety_options, default=[]
    )

with col_f3:
    if color_options:
        selected_colors = st.multiselect(
            "Color", options=color_options, default=[]
        )
    else:
        selected_colors = []

with col_f4:
    if has_year:
        selected_year_range = st.slider(
            "Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max)
        )
    else:
        selected_year_range = None


# Apply filters to the DataFrame
filter_mask = pd.Series(True, index=df.index)

# Region filter
if selected_regions and "__location_last__" in df.columns:
    filter_mask &= df["__location_last__"].astype(str).isin(selected_regions)

# Variety filter
if selected_varieties and variety_field:
    # When variety_field refers to a real column, use it directly; otherwise use the parsed column
    if variety_field in df.columns:
        filter_mask &= df[variety_field].astype(str).isin(selected_varieties)
    else:
        filter_mask &= df["__variety__"].astype(str).isin(selected_varieties)

# Color filter
if selected_colors and color_col:
    filter_mask &= df[color_col].astype(str).isin(selected_colors)

# Year slider filter
if selected_year_range and has_year:
    min_year, max_year = selected_year_range
    filter_mask &= df["__year__"].between(min_year, max_year, inclusive="both")

df_filtered = df.loc[filter_mask].copy()


# ----------------------------------------------------------------------------
# 4) Key Performance Indicators
#
# Display some simple KPIs based on the filtered data.  We show the total
# number of wines, the average rating, the median price, and either the
# average ABV or the number of unique locations.
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
#
# We keep the visualizations straightforward: horizontal bar charts for top
# categories, a line chart for average rating over time, and a histogram for
# price distribution.  Each chart is shown only when there is relevant data.
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
        top_locations.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Location")
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
    plt.hist(df_filtered[price_col].dropna(), bins=30)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig_price)
else:
    st.info("Price data not available to plot a distribution.")


# D) Top Varieties
st.subheader("Top Varieties")
# Determine which series to use for varieties
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
        top_varieties.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Variety")
        plt.tight_layout()
        st.pyplot(fig_var)
    else:
        st.info("No variety data after filters.")
else:
    st.info("Variety field not available.")


st.divider()
st.caption(
    "Built for a 15‑minute demo: explore the data with filters above, "
    "view summary KPIs, and see simple charts for locations, ratings, prices, "
    "and varieties."
)
