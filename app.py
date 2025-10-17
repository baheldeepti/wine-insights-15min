import os
import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


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
        df = pd.read_csv(local_path, encoding="utf-16", on_bad_lines="skip")
        return df, "✅ Loaded local data/vivno_dataset.csv"

    # Otherwise load from GitHub raw URL (repo root)
    github_url = "https://raw.githubusercontent.com/baheldeepti/wine-insights-15min/main/vivno_dataset.csv"
    try:
        df = pd.read_csv(github_url, encoding="utf-16", on_bad_lines="skip")
        return df, "✅ Loaded data from GitHub repository"
    except Exception:
        pass

    # fallback: KaggleHub
    kb = try_kagglehub_download()
    if kb and os.path.exists(kb):
        df = pd.read_csv(kb, encoding="utf-16", on_bad_lines="skip")
        return df, f"Loaded via KaggleHub: {kb}"

    # last resort: teaching env
    mounted = "/mnt/data/vivno_dataset.csv"
    if os.path.exists(mounted):
        df = pd.read_csv(mounted, encoding="utf-16", on_bad_lines="skip")
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

# Map likely columns (dataset-specific names included)
price_col    = map_column(df, ["Prices", "price", "wine_price", "price_usd"])
rating_col   = map_column(df, ["Ratings", "rating", "points", "average_rating"])
ratings_ncol = map_column(df, ["Ratingsnum", "ratings_count", "n_ratings"])
country_col  = map_column(df, ["Countrys", "country", "country_name", "region"])
color_col    = map_column(df, ["color_wine", "color", "wine_color"])
abv_col      = map_column(df, ["ABV %", "abv", "alcohol", "alcohol_percent"])
name_col     = map_column(df, ["Names", "name", "title"])
variety_col  = map_column(df, ["Variety", "variety", "grape", "grapes", "wine_variety"])  # NEW


# Coerce numerics
if price_col:
    df[price_col] = coerce_numeric(df[price_col])
if rating_col:
    df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
if ratings_ncol:
    df[ratings_ncol] = pd.to_numeric(df[ratings_ncol], errors="coerce")
if abv_col:
    df[abv_col] = pd.to_numeric(df[abv_col], errors="coerce")

# Derive a clean location from 'Countrys' like "Chardonnay from Sonoma County, California"
# 1) extract the part after 'from ' ; fall back to original text
df["__location__"] = (
    df[country_col].astype(str).str.extract(r"from\s+(.*)", expand=False)
    if country_col else pd.Series(np.nan, index=df.index)
)
if country_col:
    df["__location__"] = df["__location__"].fillna(df[country_col].astype(str))
# 2) keep the LAST token after comma as a short location label (e.g., "California")
df["__location_last__"] = df["__location__"].astype(str).str.split(",").str[-1].str.strip()

# Derive a simple grape/variety from 'Countrys' (text before " from ")
df["__variety__"] = (
    df[country_col].astype(str).str.extract(r"^(.*?)\s+from\s", expand=False)
    if country_col else np.nan
)
# Unified field we’ll use throughout for variety/grape (prefer real column; else parsed)
variety_field = variety_col if variety_col else "__variety__"


# Year: extract 4-digit year from Names if present
if name_col:
    df["__year__"] = pd.to_numeric(
        df[name_col].astype(str).str.extract(r"(\d{4})", expand=False),
        errors="coerce",
    )
else:
    df["__year__"] = np.nan


# Sidebar filters
st.sidebar.header("Filters")
# Region/State filter (use short label we derived)
if "__location_last__" in df.columns:
    regions = (
        df["__location_last__"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
    )
    regions = sorted(regions)
    sel_countries = st.sidebar.multiselect(  # keep variable name to minimize downstream edits
        "Region / State", options=regions, default=regions[: min(5, len(regions))]
    )
else:
    sel_countries = []


# Variety filter (prefer actual column; fallback to parsed __variety__)
if variety_field:
    varieties = (
        df[variety_field].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
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

if "__location_last__" in df.columns and sel_countries:
    filt &= df["__location_last__"].astype(str).isin(sel_countries)


if variety_field and sel_varieties:
    filt &= df[variety_field].astype(str).isin(sel_varieties)

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
    # Prefer ABV if present; else show unique short locations
    if abv_col and df_f[abv_col].notna().any():
        st.metric("Avg ABV", f"{df_f[abv_col].mean():.1f}%")
    elif "__location_last__" in df_f.columns:
        st.metric("Locations", df_f["__location_last__"].nunique())
    else:
        st.metric("Locations", "—")


st.divider()

# Charts (matplotlib for minimal deps)
# 1) Top locations by number of wines (short label)
st.subheader("Top Locations by Number of Wines")
if "__location_last__" in df_f.columns:
    top_loc = (
        df_f["__location_last__"].dropna()
        .astype(str).str.strip().replace("", np.nan).dropna()
        .value_counts().head(10).sort_values(ascending=True)
    )
    if not top_loc.empty:
        fig = plt.figure(figsize=(8, 5))
        top_loc.plot(kind="barh")
        plt.xlabel("Count")
        plt.ylabel("Location")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No location data after filters.")
else:
    st.info("Location field not available in dataset.")


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

# 4) Top grape/variety from parsed text
# 4) Top grape/variety
st.subheader("Top Varieties")
if variety_field:
    top_var = (
        df_f[variety_field].dropna()
        .astype(str).str.strip().replace("", np.nan).dropna()
        .value_counts().head(10).sort_values(ascending=True)
    )
    if not top_var.empty:
        fig4 = plt.figure(figsize=(8, 5))
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
st.caption("Tip: refine filters in the sidebar. Keep it simple, sip the insights. 🥂")
