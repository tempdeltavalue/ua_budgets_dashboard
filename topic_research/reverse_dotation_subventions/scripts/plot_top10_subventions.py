import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas as gpd
import os
import textwrap
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default')

data_dir = "article_data/data"
plots_dir = "article_data/plots"
os.makedirs(plots_dir, exist_ok=True)

print("Loading all_subventions_parsed.json...")
with open(os.path.join(data_dir, 'all_subventions_parsed.json'), 'r', encoding='utf-8') as f:
    parsed_subs = json.load(f)

print("Loading l3.geojson...")
with open('t_app/viz_dashboard_data2/l3.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
years = [2022, 2023, 2024, 2025]

# Group subventions by code to avoid duplicate entries for the same code (e.g., historical vs current names)
subventions_dict = {}
for sub in parsed_subs:
    code = sub['code']
    name = sub['name']
    
    # If code already exists, we might want to keep the name that doesn't say "виключено"
    if code in subventions_dict:
        if "виключено" in subventions_dict[code]['name'].lower() and "виключено" not in name.lower():
            subventions_dict[code]['name'] = name
    else:
        subventions_dict[code] = {'name': name}

# Calculate totals for all unique codes
subvention_totals = []

for code, data in subventions_dict.items():
    name = data['name']
    
    yearly_sums = []
    total_4_years = 0
    
    for year in years:
        col = f'INC_{code}_{year}'
        year_sum = 0
        if col in gdf.columns:
            year_sum = pd.to_numeric(gdf[col], errors='coerce').fillna(0).sum()
        yearly_sums.append(year_sum)
        total_4_years += year_sum
        
    subvention_totals.append({
        'code': code,
        'name': name,
        'yearly_sums': yearly_sums,
        'total_4_years': total_4_years
    })

# Sort by total_4_years descending and take top 10
top10_subs = sorted(subvention_totals, key=lambda x: x['total_4_years'], reverse=True)[:10]

print("Top 10 subventions by total volume:")
for i, sub in enumerate(top10_subs):
    print(f"{i+1}. [{sub['code']}] {sub['name']} - {sub['total_4_years']/1e9:.2f} млрд")

# Plotting
fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', 'x']
colors = plt.cm.tab10.colors

for i, sub in enumerate(top10_subs):
    full_name = f"[{sub['code']}] {sub['name']}"
    wrapped_name = textwrap.fill(full_name, width=60)
    
    # Convert to billions
    values_in_bln = [val / 1e9 for val in sub['yearly_sums']]
    
    ax.plot(years, values_in_bln, 
            marker=markers[i % len(markers)], 
            color=colors[i % len(colors)],
            linewidth=2.5, label=wrapped_name)

ax.set_title('ТОП-10 найбільших субвенцій за 2022-2025 роки (Сумарно)', fontsize=18, fontweight='bold')
ax.set_xlabel('Рік', fontsize=14)
ax.set_ylabel('Сума (Мільярди грн)', fontsize=14)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xticks(years)

ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plot_path = os.path.join(plots_dir, 'subventions_top10.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved plot to {plot_path}")
