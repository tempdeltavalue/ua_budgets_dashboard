import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas as gpd
import os
import json
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default')

data_dir = "article_data/data"
plots_dir = "article_data/plots"
os.makedirs(plots_dir, exist_ok=True)

print("Loading subventions CSV to get groups...")
df_groups = pd.read_csv(os.path.join(data_dir, 'all_subventions_list.csv'))

# Create a mapping from group to list of codes
group_to_codes = {}
for _, row in df_groups.iterrows():
    group = row['Група']
    code = str(row['Код'])
    
    if group == 'Історичні (Виключені)':
        continue
        
    if group not in group_to_codes:
        group_to_codes[group] = []
    group_to_codes[group].append(code)

print("Loading l3.geojson...")
with open('t_app/viz_dashboard_data2/l3.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
years = [2022, 2023, 2024, 2025]

# Calculate totals for each group per year
group_totals = {group: [] for group in group_to_codes}

for year in years:
    for group, codes in group_to_codes.items():
        year_sum = 0
        for code in codes:
            col = f'INC_{code}_{year}'
            if col in gdf.columns:
                year_sum += pd.to_numeric(gdf[col], errors='coerce').fillna(0).sum()
        
        # Convert to billions
        group_totals[group].append(year_sum / 1e9)

# Plotting
fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
markers = ['o', 's', '^', 'D', 'v', 'p', '*']
colors = plt.cm.tab10.colors

for i, (group, totals) in enumerate(group_totals.items()):
    ax.plot(years, totals, 
            marker=markers[i % len(markers)], 
            color=colors[i % len(colors)],
            linewidth=3, label=group)
            
    # Annotate the values
    for j, txt in enumerate(totals):
        if txt > 1.0: # Only annotate significant amounts to avoid clutter
            offset = 10 if i % 2 == 0 else -15
            ax.annotate(f"{txt:.1f}", (years[j], totals[j]), textcoords="offset points", 
                        xytext=(0,offset), ha='center', fontsize=9, color=colors[i % len(colors)],
                        fontweight='bold')

ax.set_title('Динаміка субвенцій за тематичними групами (2022-2025)', fontsize=18, fontweight='bold')
ax.set_xlabel('Рік', fontsize=14)
ax.set_ylabel('Сума (Мільярди грн)', fontsize=14)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xticks(years)

# Log scale might be good here as well, because "Освіта" will dwarf everything
ax.set_yscale('log')
ax.set_ylabel('Сума (Мільярди грн, логарифмічна шкала)', fontsize=14)

ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('#f8f9fa')

plt.tight_layout()
plot_path = os.path.join(plots_dir, 'subventions_by_group.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved plot to {plot_path}")
