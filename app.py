import os
import pathlib
import pandas as pd
import numpy as np
import streamlit as st

# Optional: KaggleHub fallback
def try_kagglehub_download():
    try:
        import kagglehub
        path = kagglehub.dataset_download("salohiddindev/wine-dataset-scraping-from-wine-com")
        # Try to find a likely csv file
        candidates = [
            "vivno_dataset.csv", "vivno_data.csv", "vivino_dataset.csv", "vivno_dataset (1).csv"
        ]
        for c in candidates:
            f = pathlib.Path(path) / c
            if f.exists():
                return str(f)
        # If not found by name, pick any CSV in the folder
        for f in pathlib.Path(path).glob("*.csv"):
            return str(f)
    except Exception:
        return None
    return None

# --------- Helpers: robust column mapping ----------
def map_column(df, options):
    """Return the first matching column (case-insensitive) from options, else None."""
    lower_cols = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_cols:
            return lower_cols[opt.lower()]
    return None

def coerce_numeric(s):
    """Convert strings like '$12.99', '12,99', '95 points' to numeric if possible."""
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    # strip currency and words
    cleaned = (
        s.astype(str)
         .str.replace(r"[^0-9\.\-]", "", regex=True)
         .replace({"": np.nan, ".": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")

# --------- Load data ----------
def load_data():
    # Try local path first
    local_path = "data/vivno_dataset.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, encoding="utf-8", on_bad_lines="skip")
        return df, "✅ Loaded local data/vivno_dataset.csv"

    # Otherwise load from GitHub raw URL
    github_url = "https://raw.githubusercontent.com/baheldeepti/wine-insights-15min/main/vivno_dataset.csv"
    try:
        df = pd.read_csv(github_url, encoding="utf-8", on_bad_lines="skip")
        return df, "✅ Loaded data from GitHub repository"
    except Exception as e:
        return None, f"❌ Error loading data: {e}"
  

    # fallback: KaggleHub
    kb = try_kagglehub_download()
    if kb and os.path.exists(kb):
        df = pd.read_csv(kb, encoding="utf-8", on_bad_lines="skip")
        return df, f"Loaded via KaggleHub: {kb}"

    # last resort: try a mounted path (e.g., teaching env)
    mounted = "/mnt/data/vivno_dataset.csv"
    if os.path.exists(mounted):
        df = pd.read_csv(mounted, encoding="utf-8", on_bad_lines="skip")
        return df, "Loaded /mnt/data/vivno_dataset.csv"

    return None, "No CSV found locally; and KaggleHub fallback failed."

# --------- App ---------
st.set_page_config(page_title="Wine Insights (15-min Sprint)", layout="wide")
st.title("🍷 Wine Insights — quick dashboard")

df, load_msg = load_data()
st.caption(load_msg)

if df is None or df.empty:
    st.error("Could not load any data. Please place your CSV at **data/vivno_dataset.csv**.")
    st.stop()

# Map likely columns
price_col   = map_column(df, ["price", "wine_price", "price_usd"])
rating_col  = map_column(df, ["rating", "ratings", "points", "average_rating"])
country_col = map_column(df, ["country", "country_name"])
variety_col = map_column(df, ["variety", "grape", "grapes", "wine_variety"])
year_col    = map_column(df, ["year", "vintage"])

# Coerce numerics if present
if price_col:
    df[price_col] = coerce_numeric(df[price_col])
if rating_col:
    df[rating_col] = coerce_numeric(df[rating_col])

# Try to normalize year from vintage if needed (e.g., "2018", "NV")
if year_col:
    df["__year__"] = pd.to_numeric(
        df[year_col].astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
else:
    df["__year__"] = np.nan

# Sidebar filters
st.sidebar.header("Filters")
# Country filter
if country_col:
    countries = (
        df[country_col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique()
        .tolist()
    )
    countries = sorted(countries)
    sel_countries = st.sidebar.multiselect(
        "Country", options=countries, default=countries[: min(5, len(countries))]
    )
else:
    sel_countries = []

# Variety filter
if variety_col:
    varieties = (
        df[variety_col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique()
        .tolist()
    )
    varieties = sorted(varieties)
    sel_varieties = st.sidebar.multiselect(
        "Variety / Grape", options=varieties, default=varieties[: min(5, len(varieties))]
    )
else:
    sel_varieties = []

# Year range
if df["__year__"].notna().any():
    y_min = int(df["__year__"].min(skipna=True))
    y_max = int(df["__year__"].max(skipna=True))
    yr = st.sidebar.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))
else:
    yr = None

# Apply filters
filt = pd.Series(True, index=df.index)

if country_col and sel_countries:
    filt &= df[country_col].astype(str).isin(sel_countries)

if variety_col and sel_varieties:
    filt &= df[variety_col].astype(str).isin(sel_varieties)

if yr and df["__year__"].notna().any():
    filt &= df["__year__"].between(yr[0], yr[1], inclusive="both")

df_f = df.loc[filt].copy()

st.subheader("Key KPIs")
col1, col2, col3, col4 = st.columns(4, gap="large")
with col1:
    st.metric("Total Wines", f"{len(df_f):,}")
with col2:
    if rating_col and df_f[rating_col].notna().any():
        st.metric("Avg Rating", f"{df_f[rating_col].mean():.2f}")
    else:
        st.metric("Avg Rating", "—")
with col3:
    if price_col and df_f[price_col].notna().any():
        st.metric("Median Price", f"${df_f[price_col].median():,.2f}")
    else:
        st.metric("Median Price", "—")
with col4:
    if country_col:
        st.metric("Countries", df_f[country_col].nunique())
    else:
        st.metric("Countries", "—")

st.divider()

# Charts (matplotlib for minimal deps)
import matplotlib.pyplot as plt

# 1) Top countries by count
st.subheader("Top Countries by Number of Wines")
if country_col:
    top_cty = (
        df_f[country_col].dropna()
        .astype(str)
        .value_counts()
        .head(10)
        .sort_values(ascending=True)
    )
    if not top_cty.empty:
        fig = plt.figure(figsize=(8, 5))
        top_cty.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Country")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No country data after filters.")
else:
    st.info("Country column not found in dataset.")

# 2) Average rating by year (line)
st.subheader("Average Rating by Year")
if rating_col and df_f["__year__"].notna().any():
    grp = df_f.dropna(subset=["__year__"])[["__year__", rating_col]].groupby("__year__").mean()
    if not grp.empty:
        fig2 = plt.figure(figsize=(8, 4))
        plt.plot(grp.index, grp[rating_col], marker="o")
        plt.xlabel("Year")
        plt.ylabel("Average Rating")
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("No year/rating data after filters.")
else:
    st.info("Year or rating not available to plot the trend.")

# 3) Price distribution (histogram)
st.subheader("Price Distribution")
if price_col and df_f[price_col].notna().any():
    fig3 = plt.figure(figsize=(8, 4))
    plt.hist(df_f[price_col].dropna(), bins=30)
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig3)
else:
    st.info("Price data not available to plot a distribution.")

st.divider()
st.caption("Tip: refine filters in the sidebar. Keep it simple, sip the insights. 🥂")
