import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default')

data_dir = "article_data/data"
plots_dir = "article_data/plots"
os.makedirs(plots_dir, exist_ok=True)

print("Plotting Dotations Dynamics...")
# 1. Dynamics plot
totals_csv = os.path.join(data_dir, 'dotations_totals.csv')
df_totals = pd.read_csv(totals_csv)

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
years = df_totals['Рік']

ax.plot(years, df_totals['Сума Базова'] / 1e9, marker='o', linewidth=2, label='Базова дотація')
ax.plot(years, df_totals['Сума Реверсна'] / 1e9, marker='s', linewidth=2, label='Реверсна дотація')
ax.plot(years, df_totals['Сума Додаткові'] / 1e9, marker='^', linewidth=2, label='Додаткові дотації (сума діючих)')

ax.set_title('Динаміка дотацій (тільки діючі, 2022-2025)', fontsize=14)
ax.set_xlabel('Рік', fontsize=12)
ax.set_ylabel('Сума (Мільярди грн)', fontsize=12)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xticks(years)
ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('white')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dotations_dynamics.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. Maps plotting
print("Loading l3.geojson and crimea.geojson for mapping...")
gdf = gpd.read_file('t_app/viz_dashboard_data2/l3.geojson')
gdf['BUDGET_CODE'] = gdf['BUDGET_CODE'].astype(str).str.strip()

crimea_gdf = gpd.read_file('t_app/viz_dashboard_data2/crimea.geojson')

for year in years:
    print(f"Plotting maps for {year}...")
    year_csv = os.path.join(data_dir, f'dotations_{year}.csv')
    df_year = pd.read_csv(year_csv, dtype={'Айді громади': str})
    df_year['Айді громади'] = df_year['Айді громади'].astype(str).str.strip()
    
    merged = gdf.merge(df_year, left_on='BUDGET_CODE', right_on='Айді громади', how='left')
    
    # Horizontal layout: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
    
    columns_to_plot = [
        ('Сума Базова', 'Blues', f'Базова дотація ({year})'),
        ('Сума Реверсна', 'Reds', f'Реверсна дотація ({year})'),
        ('Сума Додаткові', 'Greens', f'Сума додаткових дотацій ({year})')
    ]
    
    import matplotlib.colors as mcolors
    for i, (col, cmap, title) in enumerate(columns_to_plot):
        ax = axes[i]
        
        # Plot Crimea first
        crimea_gdf.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.1)
        
        # Plot base with light grey for missing geometries
        merged.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.1)
        
        # Plot data where it exists and is > 0
        valid_data = merged[merged[col] > 0]
        if not valid_data.empty:
            # Use LogNorm to prevent skewed data from making most regions white
            vmin = valid_data[col].min()
            vmax = valid_data[col].max()
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            
            valid_data.plot(column=col, ax=ax, cmap=cmap, legend=True, 
                            norm=norm,
                            legend_kwds={'label': 'Сума (грн)', 'orientation': 'horizontal', 'shrink': 0.8, 'pad': 0.05},
                            edgecolor='black', linewidth=0.1)
            
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'dotations_maps_{year}.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

print("Dotations plots generated.")
