# ============================================================
# NYC AIRBNB 2019 DATA ANALYSIS
# Q1 — Price variation
# Q2 — Geospatial clusters on NYC map
# Q3 — Minimum nights vs availability
# ============================================================

# If required, install these first:
# !pip install pandas numpy matplotlib seaborn geopandas shapely requests


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import geopandas as gpd
import requests


# ============================================================
# PLOT STYLE
# ============================================================

sns.set_theme(
    style="whitegrid",
    context="talk"
)


# ============================================================
# COLOR PALETTE
# ============================================================

PALETTE = {
    "Manhattan": "#6C5CE7",
    "Brooklyn": "#00B894",
    "Queens": "#FDCB6E",
    "Bronx": "#E17055",
    "Staten Island": "#0984E3"
}


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    '/mnt/data/AB_NYC_2019.csv'
)

print("=" * 70)
print("NYC AIRBNB 2019 DATA ANALYSIS")
print("=" * 70)

print(f"\nTotal listings in dataset: {len(df):,}")


# ============================================================
# DATA CLEANING
# ============================================================

# Remove listings with zero or invalid price
df_clean = df[
    df['price'] > 0
].copy()

print(
    f"Listings after removing $0 prices: "
    f"{len(df_clean):,}"
)


# Create capped price ONLY for visualization
# Original price remains unchanged
df_clean['price_capped'] = df_clean[
    'price'
].clip(upper=500)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA INFORMATION")
print("=" * 70)

print(
    df_clean[
        [
            'price',
            'minimum_nights',
            'number_of_reviews',
            'availability_365'
        ]
    ].describe()
)


# ============================================================
# QUESTION 1
# ============================================================
# How do prices vary across different neighborhoods
# and room types?
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 1")
print("=" * 70)


# ------------------------------------------------------------
# Median price by borough
# ------------------------------------------------------------

order = (
    df_clean
    .groupby('neighbourhood_group')['price']
    .median()
    .sort_values(ascending=False)
    .index
)

print("\nMedian price by borough:")

for borough in order:

    median_price = df_clean[
        df_clean['neighbourhood_group'] == borough
    ]['price'].median()

    print(
        f"{borough}: ${median_price:.0f}/night"
    )


# ------------------------------------------------------------
# Manhattan entire-home median
# ------------------------------------------------------------

manhattan_entire_median = df_clean[
    (df_clean['neighbourhood_group'] == 'Manhattan') &
    (df_clean['room_type'] == 'Entire home/apt')
]['price'].median()

print(
    f"\nManhattan entire-home median: "
    f"${manhattan_entire_median:.0f}/night"
)


# ------------------------------------------------------------
# Q1 GRAPH
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(16, 9)
)

sns.boxplot(
    data=df_clean,
    x='neighbourhood_group',
    y='price_capped',
    hue='room_type',
    order=order,
    showfliers=False,
    ax=ax,
    palette={
        "Entire home/apt": "#6C5CE7",
        "Private room": "#00B894",
        "Shared room": "#FDCB6E"
    }
)


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax.set_title(
    "Q1. How do prices vary across different neighborhoods and room types?",
    fontsize=20,
    fontweight='bold'
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel(
    "Neighborhood Group",
    fontsize=14
)

ax.set_ylabel(
    "Price per night ($)",
    fontsize=14
)


# Currency formatting
ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter(
        '${x:,.0f}'
    )
)


# ------------------------------------------------------------
# Tick formatting
# ------------------------------------------------------------

ax.tick_params(
    axis='x',
    rotation=20,
    labelsize=11
)

ax.tick_params(
    axis='y',
    labelsize=11
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    title="Room Type",
    loc='upper right'
)


# ------------------------------------------------------------
# Q1 ANSWER BOX
# ------------------------------------------------------------

answer_q1 = (
    "Answer: Manhattan has the highest price premium.\n"
    f"Entire-home median in Manhattan ≈ "
    f"${manhattan_entire_median:.0f}/night.\n"
    "Shared rooms are generally the cheapest."
)

ax.text(
    0.02,
    0.98,
    answer_q1,
    transform=ax.transAxes,
    fontsize=13,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round,pad=0.6',
        facecolor='white',
        edgecolor='black',
        linewidth=1.5
    )
)


plt.tight_layout()

plt.savefig(
    '/mnt/data/Q1_price_by_borough_room.png',
    dpi=200,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# QUESTION 2
# ============================================================
# Are there geospatial clusters of highly reviewed listings?
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 2")
print("=" * 70)


# ------------------------------------------------------------
# Find top 10% review threshold
# ------------------------------------------------------------

review_threshold = df_clean[
    'number_of_reviews'
].quantile(0.90)


# Highly reviewed listings
highly_reviewed = df_clean[
    df_clean['number_of_reviews'] >= review_threshold
].copy()


print(
    f"\nTop 10% review threshold: "
    f"{review_threshold:.0f} reviews"
)

print(
    f"Highly reviewed listings: "
    f"{len(highly_reviewed):,}"
)


# ============================================================
# DOWNLOAD OFFICIAL NYC BOROUGH MAP
# ============================================================

print("\nDownloading NYC borough boundaries...")


# NYC Open Data — Borough Boundaries
nyc_url = (
    "https://data.cityofnewyork.us/resource/"
    "gthc-hcne.geojson"
)


response = requests.get(
    nyc_url
)


if response.status_code != 200:

    raise Exception(
        "Unable to download NYC borough boundary data. "
        f"HTTP status: {response.status_code}"
    )


geojson = response.json()


# Convert GeoJSON to GeoDataFrame
nyc_map = gpd.GeoDataFrame.from_features(
    geojson["features"],
    crs="EPSG:4326"
)


print("NYC map loaded successfully.")


# ============================================================
# Q2 NYC MAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 11)
)


# ------------------------------------------------------------
# Draw NYC boroughs
# ------------------------------------------------------------

nyc_map.plot(
    ax=ax,
    color="#E6E6E6",
    edgecolor="white",
    linewidth=1.5
)


# ------------------------------------------------------------
# Plot ALL Airbnb listings
# ------------------------------------------------------------

ax.scatter(
    df_clean['longitude'],
    df_clean['latitude'],
    s=9,
    color="#6FA8DC",
    alpha=0.18,
    edgecolors='none',
    label="All listings"
)


# ------------------------------------------------------------
# Plot HIGHLY REVIEWED listings
# ------------------------------------------------------------

ax.scatter(
    highly_reviewed['longitude'],
    highly_reviewed['latitude'],
    s=22,
    color="#E67E22",
    alpha=0.65,
    edgecolors='none',
    label=(
        f"Highly reviewed "
        f"(≥ {review_threshold:.0f} reviews)"
    )
)


# ============================================================
# BOROUGH LABELS
# ============================================================

borough_labels = {

    "Manhattan": (
        -73.97,
        40.775
    ),

    "Bronx": (
        -73.86,
        40.85
    ),

    "Brooklyn": (
        -73.95,
        40.65
    ),

    "Queens": (
        -73.82,
        40.735
    ),

    "Staten Island": (
        -74.15,
        40.58
    )
}


for borough, (longitude, latitude) in borough_labels.items():

    ax.text(
        longitude,
        latitude,
        borough,
        fontsize=12,
        fontweight='bold',
        color="#444444",
        ha='center',
        va='center'
    )


# ============================================================
# Q2 TITLE
# ============================================================

ax.set_title(
    "Q2. Are there geospatial clusters of highly reviewed listings?",
    fontsize=20,
    fontweight='bold',
    pad=15
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Longitude",
    fontsize=13
)

ax.set_ylabel(
    "Latitude",
    fontsize=13
)


# ============================================================
# NYC MAP BOUNDARIES
# ============================================================

ax.set_xlim(
    -74.27,
    -73.68
)

ax.set_ylim(
    40.48,
    40.93
)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    loc='upper left',
    fontsize=11,
    frameon=True
)


# ============================================================
# Q2 ANSWER BOX
# ============================================================

answer_q2 = (
    "Answer: Yes. Highly reviewed listings are geographically "
    "concentrated,\n"
    "especially in dense Manhattan and Brooklyn Airbnb areas.\n"
    f"Top 10% threshold: ≥ {review_threshold:.0f} reviews."
)

ax.text(
    0.02,
    0.03,
    answer_q2,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='bottom',
    bbox=dict(
        boxstyle='round,pad=0.6',
        facecolor='white',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.95
    )
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    alpha=0.2
)


plt.tight_layout()


# Save Q2
plt.savefig(
    '/mnt/data/Q2_NYC_map_review_clusters.png',
    dpi=200,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# QUESTION 3
# ============================================================
# What is the correlation between minimum nights
# and listing availability?
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 3")
print("=" * 70)


# ============================================================
# PEARSON CORRELATION
# ============================================================

# IMPORTANT:
# Calculate correlation using the FULL dataset.
#
# Do NOT remove extreme minimum-night values here.

correlation = df_clean[
    ['minimum_nights', 'availability_365']
].corr().iloc[0, 1]


print(
    f"\nPearson correlation: "
    f"{correlation:.3f}"
)


# ============================================================
# INTERPRETATION
# ============================================================

if correlation >= 0.7:

    interpretation = (
        "strong positive relationship"
    )

elif correlation >= 0.3:

    interpretation = (
        "moderate positive relationship"
    )

elif correlation > 0:

    interpretation = (
        "very weak positive relationship"
    )

elif correlation <= -0.7:

    interpretation = (
        "strong negative relationship"
    )

elif correlation <= -0.3:

    interpretation = (
        "moderate negative relationship"
    )

else:

    interpretation = (
        "very weak relationship"
    )


print(
    f"Interpretation: {interpretation}"
)


# ============================================================
# PREPARE DATA FOR Q3 VISUALIZATION
# ============================================================

# The dataset contains some extreme values such as
# minimum_nights = 1250.
#
# These extreme values make the graph difficult to read.
#
# Therefore:
# - FULL DATASET → correlation
# - <= 30 nights → visualization only

plot_df = df_clean[
    df_clean['minimum_nights'] <= 30
].copy()


# ============================================================
# TREND LINE
# ============================================================

x = plot_df[
    'minimum_nights'
]

y = plot_df[
    'availability_365'
]


slope, intercept = np.polyfit(
    x,
    y,
    1
)


x_line = np.linspace(
    1,
    30,
    100
)


y_line = (
    slope * x_line +
    intercept
)


# ============================================================
# Q3 GRAPH
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 9)
)


# ------------------------------------------------------------
# Scatter plot
# ------------------------------------------------------------

ax.scatter(
    x,
    y,
    alpha=0.25,
    s=25,
    color="#2878A8",
    edgecolors='none'
)


# ------------------------------------------------------------
# Trend line
# ------------------------------------------------------------

ax.plot(
    x_line,
    y_line,
    color="#2878A8",
    linewidth=2.5
)


# ============================================================
# Q3 TITLE
# ============================================================

ax.set_title(
    f"Q3. What is the correlation between minimum nights "
    f"and listing availability?\n"
    f"(Pearson r = {correlation:.3f})",
    fontsize=20,
    fontweight='bold'
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Minimum nights",
    fontsize=14
)

ax.set_ylabel(
    "Availability (days out of 365)",
    fontsize=14
)


# ============================================================
# AXIS LIMITS
# ============================================================

ax.set_xlim(
    0,
    30
)

ax.set_ylim(
    0,
    365
)


ax.set_xticks(
    range(0, 31, 5)
)

ax.set_yticks(
    range(0, 366, 50)
)


# ============================================================
# Q3 ANSWER BOX
# ============================================================

answer_q3 = (
    f"Answer: r = {correlation:.3f}\n"
    "This is a very weak positive relationship.\n"
    "Minimum-night requirements do not strongly explain\n"
    "listing availability."
)

ax.text(
    0.02,
    0.98,
    answer_q3,
    transform=ax.transAxes,
    fontsize=13,
    verticalalignment='top',
    bbox=dict(
        boxstyle='round,pad=0.6',
        facecolor='white',
        edgecolor='black',
        linewidth=1.5
    )
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


# Save Q3
plt.savefig(
    '/mnt/data/Q3_minimum_nights_availability.png',
    dpi=200,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    "\nQ1:"
)

print(
    "Manhattan has the highest price premium."
)

print(
    f"Manhattan entire-home median: "
    f"${manhattan_entire_median:.0f}/night"
)


print(
    "\nQ2:"
)

print(
    "Highly reviewed listings are geographically concentrated."
)

print(
    f"Top 10% review threshold: "
    f"{review_threshold:.0f} reviews"
)


print(
    "\nQ3:"
)

print(
    f"Pearson correlation = "
    f"{correlation:.3f}"
)

print(
    f"Interpretation: "
    f"{interpretation}"
)


print("\n" + "=" * 70)
print("GRAPH FILES")
print("=" * 70)

print(
    "\nQ1:"
    "\n/mnt/data/Q1_price_by_borough_room.png"
)

print(
    "\nQ2:"
    "\n/mnt/data/Q2_NYC_map_review_clusters.png"
)

print(
    "\nQ3:"
    "\n/mnt/data/Q3_minimum_nights_availability.png"
)
