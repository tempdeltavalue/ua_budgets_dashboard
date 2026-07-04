import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas as gpd
import os
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default')

data_dir = "article_data/data"
plots_dir = "article_data/plots"
os.makedirs(plots_dir, exist_ok=True)

print("Loading all_subventions_parsed.json...")
with open(os.path.join(data_dir, 'all_subventions_parsed.json'), 'r', encoding='utf-8') as f:
    parsed_subs = json.load(f)

# Get unique codes for each group
state_codes = set([sub['code'] for sub in parsed_subs if sub['code'].startswith('4103')])
local_codes = set([sub['code'] for sub in parsed_subs if sub['code'].startswith('4105')])

print(f"Found {len(state_codes)} state subvention codes (4103...)")
print(f"Found {len(local_codes)} local subvention codes (4105...)")

print("Loading l3.geojson...")
with open('t_app/viz_dashboard_data2/l3.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
years = [2022, 2023, 2024, 2025]

state_totals = []
local_totals = []

for year in years:
    state_sum = 0
    local_sum = 0
    
    for code in state_codes:
        col = f'INC_{code}_{year}'
        if col in gdf.columns:
            state_sum += pd.to_numeric(gdf[col], errors='coerce').fillna(0).sum()
            
    for code in local_codes:
        col = f'INC_{code}_{year}'
        if col in gdf.columns:
            local_sum += pd.to_numeric(gdf[col], errors='coerce').fillna(0).sum()
            
    # Convert to billions
    state_totals.append(state_sum / 1e9)
    local_totals.append(local_sum / 1e9)

# Plotting
fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

ax.plot(years, state_totals, marker='o', linewidth=3, color='#1f77b4', 
        label='З державного бюджету (4103...)')
ax.plot(years, local_totals, marker='s', linewidth=3, color='#ff7f0e', 
        label='З місцевих бюджетів (4105...)')

ax.set_title('Динаміка субвенцій за джерелом походження (2022-2025)', fontsize=16, fontweight='bold')
ax.set_xlabel('Рік', fontsize=14)
ax.set_ylabel('Сума (Мільярди грн)', fontsize=14)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xticks(years)

# Add values on top of the markers
for i, txt in enumerate(state_totals):
    ax.annotate(f"{txt:.1f}", (years[i], state_totals[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11, fontweight='bold', color='#1f77b4')

for i, txt in enumerate(local_totals):
    ax.annotate(f"{txt:.1f}", (years[i], local_totals[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=11, fontweight='bold', color='#ff7f0e')

ax.legend(fontsize=12, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('#f8f9fa')

# Make y-axis start at 0
ax.set_ylim(bottom=0, top=max(max(state_totals), max(local_totals)) * 1.15)

plt.tight_layout()
plot_path = os.path.join(plots_dir, 'subventions_by_source.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved plot to {plot_path}")
