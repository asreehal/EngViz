import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"Manhattan": "#6C5CE7", "Brooklyn": "#00B894", "Queens": "#FDCB6E",
           "Bronx": "#E17055", "Staten Island": "#0984E3"}

df = pd.read_csv('/mnt/user-data/uploads/AB_NYC_2019.csv')

# --- Cleaning ---
df_clean = df[df['price'] > 0].copy()
df_clean['price_capped'] = df_clean['price'].clip(upper=500)  # for readable plots only

print(f"Rows total: {len(df)} | after removing $0 listings: {len(df_clean)}")
print(f"Listings never available (availability_365==0): {(df['availability_365']==0).sum()} "
      f"({(df['availability_365']==0).mean()*100:.1f}%)")

# ============================================================
# CHART 1 — Geospatial map: where listings are + price + demand
# ============================================================
fig, ax = plt.subplots(figsize=(11, 10))
for grp, color in PALETTE.items():
    sub = df_clean[df_clean['neighbourhood_group'] == grp]
    ax.scatter(sub['longitude'], sub['latitude'],
               s=np.clip(sub['number_of_reviews'], 3, 200) * 0.6,
               c=color, alpha=0.35, label=grp, edgecolors='none')
ax.set_title("NYC Airbnb Listings — Location, Borough & Demand\n(bubble size = number of reviews)",
             fontsize=16, fontweight='bold')
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
leg = ax.legend(title="Borough", loc='upper left', markerscale=1.5, fontsize=11)
for lh in leg.legend_handles:
    lh.set_alpha(1); lh.set_sizes([80])
plt.tight_layout()
plt.savefig('/home/claude/chart1_geomap.png', dpi=150)
plt.close()

# ============================================================
# CHART 2 — Price distribution by borough & room type
# ============================================================
order = df_clean.groupby('neighbourhood_group')['price'].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(12, 7))
sns.boxplot(data=df_clean, x='neighbourhood_group', y='price_capped', hue='room_type',
            order=order, showfliers=False, ax=ax,
            palette={"Entire home/apt": "#6C5CE7", "Private room": "#00B894", "Shared room": "#FDCB6E"})
ax.set_title("Price Distribution by Borough & Room Type\n(prices capped at $500 for readability)",
             fontsize=16, fontweight='bold')
ax.set_xlabel("Borough"); ax.set_ylabel("Price per night ($)")
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
plt.legend(title="Room Type", loc='upper right')
plt.tight_layout()
plt.savefig('/home/claude/chart2_price_by_borough.png', dpi=150)
plt.close()

# ============================================================
# CHART 3 — Price vs demand (does higher price = more reviews?)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))
sample = df_clean.sample(min(8000, len(df_clean)), random_state=42)
sc = ax.scatter(sample['price'], sample['number_of_reviews'],
                 c=sample['neighbourhood_group'].map(PALETTE), alpha=0.35, s=25, edgecolors='none')
ax.set_xscale('log')
ax.set_title("Price vs. Number of Reviews\n(higher price does not mean more demand)",
             fontsize=16, fontweight='bold')
ax.set_xlabel("Price per night ($, log scale)"); ax.set_ylabel("Number of reviews")
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=b)
           for b, c in PALETTE.items()]
ax.legend(handles=handles, title="Borough", loc='upper right')
plt.tight_layout()
plt.savefig('/home/claude/chart3_price_vs_reviews.png', dpi=150)
plt.close()

corr = df_clean[['price', 'number_of_reviews']].corr().iloc[0, 1]
print(f"Correlation price vs number_of_reviews: {corr:.3f}")

# ============================================================
# CHART 4 — Top 15 hosts by listing count (market concentration)
# ============================================================
host_counts = (df_clean.groupby(['host_id', 'host_name'])['id'].count()
               .reset_index(name='listing_count')
               .sort_values('listing_count', ascending=False).head(15))
host_counts['label'] = host_counts['host_name'].fillna('Unknown') + " (" + host_counts['host_id'].astype(str) + ")"

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(host_counts['label'][::-1], host_counts['listing_count'][::-1], color="#6C5CE7")
ax.set_title("Top 15 Hosts by Number of Listings\n(market concentration among 'power hosts')",
             fontsize=16, fontweight='bold')
ax.set_xlabel("Number of listings")
for bar in bars:
    w = bar.get_width()
    ax.text(w + 3, bar.get_y() + bar.get_height()/2, f"{int(w)}", va='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/claude/chart4_top_hosts.png', dpi=150)
plt.close()

top_host_share = host_counts['listing_count'].sum() / len(df_clean) * 100
print(f"Top 15 hosts alone control {top_host_share:.1f}% of all listings")

# ============================================================
# CHART 5 — Availability: how much "listed" inventory is actually inactive
# ============================================================
fig, ax = plt.subplots(figsize=(11, 7))
ax.hist(df_clean['availability_365'], bins=40, color="#00B894", edgecolor='white', alpha=0.85)
never_avail = (df_clean['availability_365'] == 0).sum()
pct = never_avail / len(df_clean) * 100
ax.axvline(0, color='#E17055', linestyle='--', linewidth=2)
ax.annotate(f"{never_avail:,} listings ({pct:.1f}%)\nnever available all year",
            xy=(5, ax.get_ylim()[1]*0.85), fontsize=13, color='#E17055', fontweight='bold')
ax.set_title("Listing Availability Over the Year\n(a large share of 'active' listings are effectively dormant)",
             fontsize=16, fontweight='bold')
ax.set_xlabel("Days available out of 365"); ax.set_ylabel("Number of listings")
plt.tight_layout()
plt.savefig('/home/claude/chart5_availability.png', dpi=150)
plt.close()

print("\nAll 5 individual charts saved.")

# ============================================================
# COMBINED DASHBOARD — all 5 charts + KPI summary in one image
# ============================================================
fig = plt.figure(figsize=(22, 13))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

fig.suptitle("NYC Airbnb Market — Where the Listed Inventory Isn't What It Looks Like",
             fontsize=22, fontweight='bold', y=0.99)

# --- Panel 1: geomap ---
ax1 = fig.add_subplot(gs[0, 0])
for grp, color in PALETTE.items():
    sub = df_clean[df_clean['neighbourhood_group'] == grp]
    ax1.scatter(sub['longitude'], sub['latitude'],
                s=np.clip(sub['number_of_reviews'], 3, 200) * 0.4,
                c=color, alpha=0.3, label=grp, edgecolors='none')
ax1.set_title("Listings by Borough & Demand", fontsize=14, fontweight='bold')
ax1.set_xlabel("Longitude", fontsize=10); ax1.set_ylabel("Latitude", fontsize=10)
ax1.legend(fontsize=8, loc='upper left', markerscale=0.8)
ax1.tick_params(labelsize=9)

# --- Panel 2: price by borough/room ---
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(data=df_clean, x='neighbourhood_group', y='price_capped', hue='room_type',
            order=order, showfliers=False, ax=ax2,
            palette={"Entire home/apt": "#6C5CE7", "Private room": "#00B894", "Shared room": "#FDCB6E"})
ax2.set_title("Price by Borough & Room Type", fontsize=14, fontweight='bold')
ax2.set_xlabel(""); ax2.set_ylabel("Price ($)", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
ax2.tick_params(axis='x', rotation=20, labelsize=9)
ax2.legend(fontsize=7, title_fontsize=8)

# --- Panel 3: price vs reviews ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(sample['price'], sample['number_of_reviews'],
            c=sample['neighbourhood_group'].map(PALETTE), alpha=0.3, s=15, edgecolors='none')
ax3.set_xscale('log')
ax3.set_title(f"Price vs. Reviews (corr = {corr:.2f})", fontsize=14, fontweight='bold')
ax3.set_xlabel("Price ($, log)", fontsize=10); ax3.set_ylabel("Reviews", fontsize=10)
ax3.tick_params(labelsize=9)

# --- Panel 4: top hosts ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.barh(host_counts['label'][::-1][-10:], host_counts['listing_count'][::-1][-10:] if False else host_counts['listing_count'][:10][::-1], color="#6C5CE7")
ax4.set_yticks(range(10))
ax4.set_yticklabels(host_counts['host_name'].fillna('Unknown').head(10)[::-1], fontsize=9)
ax4.set_title("Top 10 Hosts by Listings", fontsize=14, fontweight='bold')
ax4.set_xlabel("Listings", fontsize=10)
ax4.tick_params(labelsize=9)

# --- Panel 5: availability ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df_clean['availability_365'], bins=40, color="#00B894", edgecolor='white', alpha=0.85)
ax5.axvline(0, color='#E17055', linestyle='--', linewidth=2)
ax5.set_title(f"Availability — {pct:.0f}% Never Available", fontsize=14, fontweight='bold')
ax5.set_xlabel("Days available / 365", fontsize=10); ax5.set_ylabel("Listings", fontsize=10)
ax5.tick_params(labelsize=9)

# --- Panel 6: KPI summary text ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
kpi_text = (
    f"KEY FINDINGS\n\n"
    f"Total listings analyzed:  {len(df_clean):,}\n\n"
    f"Median price/night:  ${df_clean['price'].median():.0f}\n\n"
    f"Listings never available all year:\n   {never_avail:,}  ({pct:.1f}%)\n\n"
    f"Price \u2194 demand correlation:\n   {corr:.3f}  (essentially none)\n\n"
    f"Top host by volume:\n   {host_counts.iloc[0]['host_name']}  "
    f"({host_counts.iloc[0]['listing_count']} listings)\n\n"
    f"Manhattan share of listings:\n   "
    f"{(df_clean['neighbourhood_group']=='Manhattan').mean()*100:.1f}%"
)
ax6.text(0.02, 0.98, kpi_text, transform=ax6.transAxes, fontsize=13,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F3FF', edgecolor='#6C5CE7', linewidth=1.5))

plt.savefig('/home/claude/airbnb_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Combined dashboard saved -> airbnb_dashboard.png")
