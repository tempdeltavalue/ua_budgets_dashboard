import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import numpy as np
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default')

data_dir = "article_data/data"
plots_dir = "article_data/plots"
os.makedirs(plots_dir, exist_ok=True)

print("Loading data...")
df_res = pd.read_csv(os.path.join(data_dir, 'clustering_results.csv'), dtype={'Код громади': str, 'Рік': str})
df_res['Код громади'] = df_res['Код громади'].str.strip()

print("Loading geojson...")
gdf = gpd.read_file('t_app/viz_dashboard_data2/l3.geojson')
gdf['BUDGET_CODE'] = gdf['BUDGET_CODE'].astype(str).str.strip()
crimea_gdf = gpd.read_file('t_app/viz_dashboard_data2/crimea.geojson')

import matplotlib.cm as cm
# Use tab10 for up to 10 distinct colors
base_colors = [cm.colors.to_hex(cm.tab10(i)) for i in range(10)]

def get_color_with_alpha(cluster, distance, max_dist, min_dist):
    if pd.isna(distance) or max_dist == min_dist:
        alpha = 1.0
    else:
        norm_dist = (distance - min_dist) / (max_dist - min_dist)
        alpha = 1.0 - (norm_dist * 0.7)
        
    base_hex = base_colors[cluster % len(base_colors)]
    return to_rgba(base_hex, alpha)

def create_legend_patches(unique_clusters, prefix="Кластер"):
    patches = []
    for c in sorted(unique_clusters):
        if pd.isna(c): continue
        c = int(c)
        base_hex = base_colors[c % len(base_colors)]
        patches.append(mpatches.Patch(color=base_hex, label=f'{prefix} {c}'))
    return patches

years = [2022, 2023, 2024, 2025]

def plot_pooled(model_name, output_filename):
    print(f"Plotting {model_name}...")
    df_pooled = df_res[df_res['Вид кластеризації'] == model_name]
    if df_pooled.empty:
        print(f"No data for {model_name}")
        return
        
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), facecolor='white')
    
    for i, year in enumerate(years):
        ax = axes[i]
        crimea_gdf.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.1)
        gdf.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.1)
        
        df_y = df_pooled[df_pooled['Рік'] == str(year)]
        df_y = df_y.drop_duplicates(subset=['Код громади'])
            
        merged = gdf.merge(df_y, left_on='BUDGET_CODE', right_on='Код громади', how='left')
        gdf_plot = merged[merged['Кластер'].notna()]
        
        for c in gdf_plot['Кластер'].unique():
            subset = gdf_plot[gdf_plot['Кластер'] == c]
            color = get_color_with_alpha(int(c), 0, 1, 1) # Solid color
            subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.1)
            
        ax.set_title(f'Рік {year}', fontsize=12)
        ax.axis('off')

    plt.suptitle(f"Кластеризація: {model_name}", fontsize=16)
    unique_clusters = df_pooled['Кластер'].dropna().unique()
    patches = create_legend_patches(unique_clusters, prefix="Кластер")
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=len(patches), fontsize=14)
    
    plt.subplots_adjust(bottom=0.2)
    plot_path = os.path.join(plots_dir, output_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {plot_path}")

def plot_trajectories(model_name, output_filename):
    print(f"Plotting {model_name}...")
    df_traj = df_res[df_res['Вид кластеризації'] == model_name]
    if df_traj.empty:
        print(f"No data for {model_name}")
        return
        
    fig, ax_dtw = plt.subplots(1, 1, figsize=(10, 8), facecolor='white')
    crimea_gdf.plot(ax=ax_dtw, color='lightgrey', edgecolor='black', linewidth=0.1)
    gdf.plot(ax=ax_dtw, color='lightgrey', edgecolor='black', linewidth=0.1)

    merged_dtw = gdf.merge(df_traj, left_on='BUDGET_CODE', right_on='Код громади', how='left')
    gdf_plot_dtw = merged_dtw[merged_dtw['Кластер'].notna()]

    for c in gdf_plot_dtw['Кластер'].unique():
        subset = gdf_plot_dtw[gdf_plot_dtw['Кластер'] == c]
        color = get_color_with_alpha(int(c), 0, 1, 1)
        subset.plot(ax=ax_dtw, color=color, edgecolor='black', linewidth=0.1)

    ax_dtw.axis('off')
    ax_dtw.set_title(f"Кластеризація: {model_name} за 2022-2025", fontsize=18)

    unique_clusters = df_traj['Кластер'].dropna().unique()
    patches = create_legend_patches(unique_clusters, prefix="Кластер")
    ax_dtw.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=min(len(patches), 5), fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, output_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {plot_path}")

plot_pooled('Пул (KMeans)', 'clustering_map_pooled_kmeans.png')
plot_pooled('Пул (Математичний)', 'clustering_map_pooled_agg.png')
plot_trajectories('Траєкторії (KMeans)', 'clustering_map_traj_kmeans.png')
plot_trajectories('Траєкторії (Математичний DTW)', 'clustering_map_traj_agg.png')

print("All tasks completed.")
