import json
import geopandas as gpd
import pandas as pd
import os

data_dir = "article_data/data"
os.makedirs(data_dir, exist_ok=True)

print("Loading subventions.txt...")
df_sub = pd.read_csv('article_data/subventions.txt')
df_sub['Код'] = df_sub['Код'].astype(str).str.strip()
df_sub['Категорія'] = df_sub['Категорія'].str.strip()

print("Loading l3.geojson...")
gdf = gpd.read_file('t_app/viz_dashboard_data2/l3.geojson')
print(f"Loaded {len(gdf)} hromadas.")

years = sorted(df_sub['Рік'].unique())

totals_list = []
categories = df_sub['Категорія'].unique()

for year in years:
    print(f"Parsing subventions for {year}...")
    
    active_subs = df_sub[(df_sub['Рік'] == year) & (df_sub['Діюча'] == 1)]
    
    hromada_data = {
        'Айді громади': gdf['BUDGET_CODE'],
        'Назва громади': gdf['display_name']
    }
    
    year_totals = {'Рік': year}
    category_sums = {cat: pd.Series(0.0, index=gdf.index) for cat in categories}
    
    for _, row in active_subs.iterrows():
        cat = row['Категорія']
        code = row['Код']
        name = row['Найменування']
        
        col = f'INC_{code}_{year}'
            
        if col in gdf.columns:
            values = pd.to_numeric(gdf[col], errors='coerce').fillna(0)
        else:
            values = pd.Series(0.0, index=gdf.index)
            
        col_name = f"[{code}] {name}"
        hromada_data[col_name] = values
        year_totals[col_name] = values.sum()
        
        if cat in category_sums:
            category_sums[cat] += values
            
    # Add category sums to hromada data
    for cat in categories:
        hromada_data[f'Сума {cat}'] = category_sums[cat]
        year_totals[f'Сума {cat}'] = category_sums[cat].sum()
        
    df_hromada = pd.DataFrame(hromada_data)
    csv_path = os.path.join(data_dir, f'subventions_{year}.csv')
    df_hromada.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    totals_list.append(year_totals)

df_totals = pd.DataFrame(totals_list)
totals_csv = os.path.join(data_dir, 'subventions_totals.csv')
df_totals.to_csv(totals_csv, index=False, encoding='utf-8-sig')
print("Subventions parsed successfully.")
