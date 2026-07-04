import os
import pandas as pd
import numpy as np

data_dir = "article_data/data"
years = [2022, 2023, 2024, 2025]
features = [
    'Сума Базова', 'Сума Реверсна', 'Сума Додаткові',
    'Сума Війна та відновлення', 'Сума Освіта', "Сума Охорона здоров'я"
]

print("Loading data...")
df_res = pd.read_csv(os.path.join(data_dir, 'clustering_results.csv'), dtype={'Код громади': str})
df_res['Код громади'] = df_res['Код громади'].str.strip()

# We need to rebuild the dataset to get the actual feature values
# Let's just read the master list and then the yearly files
gdf = pd.read_csv(os.path.join(data_dir, 'dotations_2022.csv'), dtype={'Айді громади': str}) # just to get an idea
# Actually, since we want to know what each cluster means, let's load all data

master_hromadas = df_res[['Код громади', 'Назва громади']].drop_duplicates().copy()
master_hromadas.columns = ['Айді громади', 'Назва громади']

pooled_rows = []
for year in years:
    dot_file = os.path.join(data_dir, f'dotations_{year}.csv')
    sub_file = os.path.join(data_dir, f'subventions_{year}.csv')
    
    df_base = master_hromadas.copy()
    
    if os.path.exists(dot_file) and os.path.exists(sub_file):
        df_dot = pd.read_csv(dot_file, dtype={'Айді громади': str})
        df_sub = pd.read_csv(sub_file, dtype={'Айді громади': str})
        
        df_dot['Айді громади'] = df_dot['Айді громади'].astype(str).str.strip()
        df_sub['Айді громади'] = df_sub['Айді громади'].astype(str).str.strip()
        
        df_merged = pd.merge(df_base, df_dot, on='Айді громади', how='left')
        df_merged = pd.merge(df_merged, df_sub, on='Айді громади', how='left', suffixes=('', '_sub'))
        
        df_merged[features] = df_merged[features].fillna(0.0)
        df_merged['Рік'] = str(year)
        pooled_rows.append(df_merged)

df_all = pd.concat(pooled_rows, ignore_index=True)

# Merge with clustering results
# For Pooled
df_pooled_res = df_res[df_res['Вид кластеризації'] == "Об'єднаний пул (Кластери)"]
df_pooled_res['Рік'] = df_pooled_res['Рік'].astype(str)

df_pooled_res = df_pooled_res.drop(columns=['Назва громади'])
df_analysis_pooled = pd.merge(df_all, df_pooled_res, left_on=['Айді громади', 'Рік'], right_on=['Код громади', 'Рік'], how='inner')

print("\n--- Профілі кластерів для Об'єднаного пулу ---")
profile_pooled = df_analysis_pooled.groupby('Кластер')[features].mean().round(0)
profile_pooled['Кількість громад*років'] = df_analysis_pooled.groupby('Кластер').size()

def generate_description(row):
    desc = []
    if row['Сума Реверсна'] > 1e7: desc.append("Високий реверс (донори)")
    elif row['Сума Реверсна'] > 1e6: desc.append("Середній реверс")
    
    if row['Сума Базова'] > 1e7: desc.append("Висока базова дотація")
    elif row['Сума Базова'] > 1e6: desc.append("Середня базова дотація")
    elif row['Сума Базова'] == 0: desc.append("Без базової дотації")
    
    if row['Сума Війна та відновлення'] > 1e7: desc.append("Фокус на відновлення (великі суми)")
    if row['Сума Освіта'] < 1e6: desc.append("Майже нульова освіта (можливо окупація)")
    
    return " | ".join(desc) if desc else "Стандартна субсидована громада"

profile_pooled['Авто-Опис'] = profile_pooled.apply(generate_description, axis=1)
print(profile_pooled.to_string())
profile_pooled.to_csv(os.path.join(data_dir, 'cluster_profiles_pooled.csv'))

# For DTW
df_dtw_res = df_res[df_res['Вид кластеризації'] == "Траєкторії (DTW Кластери)"]
df_hromada_avg = df_all.groupby('Айді громади')[features].mean().reset_index()
df_dtw_res = df_dtw_res.drop(columns=['Назва громади'])
df_analysis_dtw = pd.merge(df_hromada_avg, df_dtw_res, left_on='Айді громади', right_on='Код громади', how='inner')

print("\n--- Профілі кластерів для Траєкторій (DTW) ---")
profile_dtw = df_analysis_dtw.groupby('Кластер')[features].mean().round(0)
profile_dtw['Кількість громад'] = df_analysis_dtw.groupby('Кластер').size()
profile_dtw['Авто-Опис'] = profile_dtw.apply(generate_description, axis=1)
print(profile_dtw.to_string())
profile_dtw.to_csv(os.path.join(data_dir, 'cluster_profiles_dtw.csv'))

print("\nПрофілі та їх автоматичний опис збережено у папку data (cluster_profiles_pooled.csv та cluster_profiles_dtw.csv)")
