# ============================================================
# NYC AIRBNB 2019 DATA ANALYSIS — GOOGLE COLAB
# Q1: Price variation
# Q2: Highly reviewed listings on NYC map
# Q3: Minimum nights vs availability
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import geopandas as gpd
import requests

sns.set_theme(style="whitegrid", context="talk")


# ------------------------------------------------------------
# 2. UPLOAD CSV
# ------------------------------------------------------------

from google.colab import files

uploaded = files.upload()

# Automatically get uploaded filename
filename = list(uploaded.keys())[0]

df = pd.read_csv(filename)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print(f"File: {filename}")
print(f"Total listings: {len(df):,}")


# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------

df_clean = df[df['price'] > 0].copy()

# Price capped ONLY for visualization
df_clean['price_capped'] = df_clean['price'].clip(upper=500)

print(f"Listings after cleaning: {len(df_clean):,}")


# ============================================================
# QUESTION 1
# ============================================================
# How do prices vary across different neighborhoods
# and room types?
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 1")
print("=" * 60)

# Borough order based on median price
order = (
    df_clean
    .groupby('neighbourhood_group')['price']
    .median()
    .sort_values(ascending=False)
    .index
)

# Manhattan entire-home median
manhattan_median = df_clean[
    (df_clean['neighbourhood_group'] == 'Manhattan') &
    (df_clean['room_type'] == 'Entire home/apt')
]['price'].median()

print("\nMedian price by borough:")

for borough in order:
    median = df_clean[
        df_clean['neighbourhood_group'] == borough
    ]['price'].median()

    print(f"{borough}: ${median:.0f}/night")

print(
    f"\nManhattan entire-home median: "
    f"${manhattan_median:.0f}/night"
)


# ------------------------------------------------------------
# Q1 PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 9))

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

ax.set_title(
    "Q1. How do prices vary across different neighborhoods and room types?",
    fontsize=20,
    fontweight='bold'
)

ax.set_xlabel("Neighborhood Group", fontsize=14)
ax.set_ylabel("Price per night ($)", fontsize=14)

ax.yaxis.set_major_formatter(
    mticker.StrMethodFormatter('${x:,.0f}')
)

ax.tick_params(axis='x', rotation=20)

ax.legend(
    title="Room Type",
    loc='upper right'
)

answer_q1 = (
    "Answer: Manhattan has the highest price premium.\n"
    f"Entire-home median in Manhattan ≈ ${manhattan_median:.0f}/night.\n"
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
plt.show()


# ============================================================
# QUESTION 2
# ============================================================
# Are there geospatial clusters of highly reviewed listings?
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 2")
print("=" * 60)


# ------------------------------------------------------------
# Find highly reviewed listings
# ------------------------------------------------------------

review_threshold = df_clean[
    'number_of_reviews'
].quantile(0.90)

highly_reviewed = df_clean[
    df_clean['number_of_reviews'] >= review_threshold
].copy()

print(
    f"Top 10% review threshold: "
    f"{review_threshold:.0f} reviews"
)

print(
    f"Highly reviewed listings: "
    f"{len(highly_reviewed):,}"
)


# ------------------------------------------------------------
# Download NYC borough boundaries
# ------------------------------------------------------------

nyc_url = (
    "https://data.cityofnewyork.us/resource/"
    "gthc-hcne.geojson"
)

response = requests.get(nyc_url)

if response.status_code != 200:
    raise Exception(
        f"Could not download NYC map. "
        f"Status code: {response.status_code}"
    )

nyc_geojson = response.json()

nyc_map = gpd.GeoDataFrame.from_features(
    nyc_geojson["features"],
    crs="EPSG:4326"
)


# ------------------------------------------------------------
# Q2 NYC MAP
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 11))

# NYC boroughs
nyc_map.plot(
    ax=ax,
    color="#E6E6E6",
    edgecolor="white",
    linewidth=1.5
)

# All Airbnb listings
ax.scatter(
    df_clean['longitude'],
    df_clean['latitude'],
    s=8,
    color="#6FA8DC",
    alpha=0.18,
    edgecolors='none',
    label="All listings"
)

# Highly reviewed listings
ax.scatter(
    highly_reviewed['longitude'],
    highly_reviewed['latitude'],
    s=22,
    color="#E67E22",
    alpha=0.65,
    edgecolors='none',
    label=f"Highly reviewed (≥ {review_threshold:.0f} reviews)"
)


# ------------------------------------------------------------
# Borough labels
# ------------------------------------------------------------

borough_labels = {
    "Manhattan": (-73.97, 40.775),
    "Bronx": (-73.86, 40.85),
    "Brooklyn": (-73.95, 40.65),
    "Queens": (-73.82, 40.735),
    "Staten Island": (-74.15, 40.58)
}

for borough, (lon, lat) in borough_labels.items():

    ax.text(
        lon,
        lat,
        borough,
        fontsize=12,
        fontweight='bold',
        color="#444444",
        ha='center'
    )


# ------------------------------------------------------------
# Q2 title
# ------------------------------------------------------------

ax.set_title(
    "Q2. Are there geospatial clusters of highly reviewed listings?",
    fontsize=20,
    fontweight='bold',
    pad=15
)

ax.set_xlabel("Longitude", fontsize=13)
ax.set_ylabel("Latitude", fontsize=13)


# ------------------------------------------------------------
# NYC map limits
# ------------------------------------------------------------

ax.set_xlim(-74.27, -73.68)
ax.set_ylim(40.48, 40.93)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    loc='upper left',
    fontsize=11
)


# ------------------------------------------------------------
# Q2 answer
# ------------------------------------------------------------

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

ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()


# ============================================================
# QUESTION 3
# ============================================================
# What is the correlation between minimum nights
# and listing availability?
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 3")
print("=" * 60)


# ------------------------------------------------------------
# Pearson correlation
# FULL DATASET
# ------------------------------------------------------------

correlation = df_clean[
    ['minimum_nights', 'availability_365']
].corr().iloc[0, 1]

print(
    f"\nPearson correlation: {correlation:.3f}"
)


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

if correlation > 0.7:
    interpretation = "strong positive relationship"
elif correlation > 0.3:
    interpretation = "moderate positive relationship"
elif correlation > 0:
    interpretation = "very weak positive relationship"
elif correlation < -0.7:
    interpretation = "strong negative relationship"
elif correlation < -0.3:
    interpretation = "moderate negative relationship"
else:
    interpretation = "very weak relationship"

print(
    f"Interpretation: {interpretation}"
)


# ------------------------------------------------------------
# Data for visualization
# ------------------------------------------------------------
# Extreme minimum-night values are excluded ONLY from
# the visualization so that the graph remains readable.
#
# The correlation above still uses the FULL dataset.

plot_df = df_clean[
    df_clean['minimum_nights'] <= 30
].copy()


# ------------------------------------------------------------
# Regression / trend line
# ------------------------------------------------------------

x = plot_df['minimum_nights']
y = plot_df['availability_365']

slope, intercept = np.polyfit(x, y, 1)

x_line = np.linspace(1, 30, 100)
y_line = slope * x_line + intercept


# ------------------------------------------------------------
# Q3 PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(15, 9))

# Scatter
ax.scatter(
    x,
    y,
    alpha=0.25,
    s=25,
    color="#2878A8",
    edgecolors='none'
)

# Trend line
ax.plot(
    x_line,
    y_line,
    color="#2878A8",
    linewidth=2.5
)


# ------------------------------------------------------------
# Q3 title
# ------------------------------------------------------------

ax.set_title(
    f"Q3. What is the correlation between minimum nights "
    f"and listing availability?\n"
    f"(Pearson r = {correlation:.3f})",
    fontsize=20,
    fontweight='bold'
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel(
    "Minimum nights",
    fontsize=14
)

ax.set_ylabel(
    "Availability (days out of 365)",
    fontsize=14
)


# ------------------------------------------------------------
# Axis limits
# ------------------------------------------------------------

ax.set_xlim(0, 30)
ax.set_ylim(0, 365)

ax.set_xticks(range(0, 31, 5))
ax.set_yticks(range(0, 366, 50))


# ------------------------------------------------------------
# Answer box
# ------------------------------------------------------------

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

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"\nQ1: Manhattan has the highest price premium."
)

print(
    f"    Manhattan entire-home median: "
    f"${manhattan_median:.0f}/night"
)

print(
    "\nQ2: Highly reviewed listings are geographically "
    "concentrated."
)

print(
    f"    Top 10% threshold: "
    f"{review_threshold:.0f} reviews"
)

print(
    f"\nQ3: Pearson correlation = "
    f"{correlation:.3f}"
)

print(
    f"    Interpretation: {interpretation}"
)

print("\nAll three graphs have been generated successfully.")
