import json
import geopandas as gpd
import pandas as pd
import os

data_dir = "article_data/data"
os.makedirs(data_dir, exist_ok=True)

print("Loading dotations.txt...")
df_dot = pd.read_csv('article_data/dotations.txt')
df_dot['Код'] = df_dot['Код'].astype(str).str.strip()

print("Loading l3.geojson...")
gdf = gpd.read_file('t_app/viz_dashboard_data2/l3.geojson')
print(f"Loaded {len(gdf)} hromadas.")

years = sorted(df_dot['Рік'].unique())

totals_list = []

for year in years:
    print(f"Parsing dotations for {year}...")
    
    # Filter active only
    active_dots = df_dot[(df_dot['Рік'] == year) & (df_dot['Діюча'] == 1)]
    
    hromada_data = {
        'Айді громади': gdf['BUDGET_CODE'],
        'Назва громади': gdf['display_name']
    }
    
    year_totals = {'Рік': year}
    category_sums = {'Базова': pd.Series(0.0, index=gdf.index), 
                     'Реверсивна': pd.Series(0.0, index=gdf.index), 
                     'Додаткові': pd.Series(0.0, index=gdf.index)}
    
    for _, row in active_dots.iterrows():
        cat = row['Категорія']
        code = row['Код']
        name = row['Найменування']
        
        if 'Реверсна' in name or 'Реверсивна' in cat or code == '41010100':
            col = f'PROG_9110_{year}'
        else:
            col = f'INC_{code}_{year}'
            
        if col in gdf.columns:
            values = pd.to_numeric(gdf[col], errors='coerce').fillna(0)
        else:
            values = pd.Series(0.0, index=gdf.index)
            
        col_name = f"[{code}] {name}"
        hromada_data[col_name] = values
        year_totals[col_name] = values.sum()
        
        # Add to category sums
        if cat in category_sums:
            category_sums[cat] += values
            
    # Add category sums to hromada data
    hromada_data['Сума Базова'] = category_sums['Базова']
    hromada_data['Сума Реверсна'] = category_sums['Реверсивна']
    hromada_data['Сума Додаткові'] = category_sums['Додаткові']
    
    year_totals['Сума Базова'] = category_sums['Базова'].sum()
    year_totals['Сума Реверсна'] = category_sums['Реверсивна'].sum()
    year_totals['Сума Додаткові'] = category_sums['Додаткові'].sum()
    
    df_hromada = pd.DataFrame(hromada_data)
    csv_path = os.path.join(data_dir, f'dotations_{year}.csv')
    df_hromada.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    totals_list.append(year_totals)

df_totals = pd.DataFrame(totals_list)
totals_csv = os.path.join(data_dir, 'dotations_totals.csv')
df_totals.to_csv(totals_csv, index=False, encoding='utf-8-sig')
print("Dotations parsed successfully.")
