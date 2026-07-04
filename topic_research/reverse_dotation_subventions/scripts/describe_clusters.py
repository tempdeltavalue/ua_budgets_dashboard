import os
import pandas as pd
import numpy as np
import joblib

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'models')
data_dir = os.path.join(base_dir, 'data')

# Load original data features to know column order
import json
master_hromadas_path = os.path.join(base_dir, '..', 't_app', 'viz_dashboard_data2', 'l3.geojson')
with open(master_hromadas_path, 'r', encoding='utf-8') as f:
    geojson = json.load(f)
master_hromadas = pd.DataFrame([f['properties'] for f in geojson['features']])
master_hromadas = master_hromadas.rename(columns={'BUDGET_CODE': 'Айді громади', 'NAME': 'Назва громади'})

features = ['Сума Базова', 'Сума Реверсна', 'Сума Додаткові', 'Сума Війна та відновлення', 'Сума Освіта', "Сума Охорона здоров'я"]
years = [2022, 2023, 2024, 2025]
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
        X = df_merged[['Айді громади'] + features].copy()
        X[features] = X[features].fillna(0.0)
        X['Рік'] = str(year)
        pooled_rows.append(X)
        
df_all = pd.concat(pooled_rows, ignore_index=True)

# Load Models and Results
print("Loading clustering results...")
df_res = pd.read_csv(os.path.join(data_dir, 'clustering_results.csv'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

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

# 1. Pooled KMeans (From Centroids)
print("\n--- Профілі кластерів: Пул (KMeans) ---")
kmeans_pooled = joblib.load(os.path.join(models_dir, 'kmeans_pooled_model.pkl'))
centers_scaled = kmeans_pooled.cluster_centers_
centers_raw = np.expm1(scaler.inverse_transform(centers_scaled))
df_pk = pd.DataFrame(centers_raw, columns=features).round(0)
df_pk.index.name = 'Кластер'
df_pk['Авто-Опис'] = df_pk.apply(generate_description, axis=1)
print(df_pk.to_string())
df_pk.to_csv(os.path.join(data_dir, 'cluster_meanings_pooled_kmeans.csv'))

# 2. Pooled Agglomerative (From Empirical Means)
print("\n--- Профілі кластерів: Пул (Математичний) ---")
df_pa_res = df_res[df_res['Вид кластеризації'] == 'Пул (Математичний)'].copy()
df_pa_res['Рік'] = df_pa_res['Рік'].astype(str)
df_all['Рік'] = df_all['Рік'].astype(str)
df_all['Айді громади'] = df_all['Айді громади'].astype(str)
df_pa_res['Код громади'] = df_pa_res['Код громади'].astype(str)

df_pa_merged = pd.merge(df_all, df_pa_res, left_on=['Айді громади', 'Рік'], right_on=['Код громади', 'Рік'], how='inner')
profile_pa = df_pa_merged.groupby('Кластер')[features].mean().round(0)
profile_pa['Кількість спостережень'] = df_pa_merged.groupby('Кластер').size()
profile_pa['Авто-Опис'] = profile_pa.apply(generate_description, axis=1)
print(profile_pa.to_string())
profile_pa.to_csv(os.path.join(data_dir, 'cluster_meanings_pooled_agg.csv'))

# 3. Trajectories KMeans (From 24D Centroids)
print("\n--- Профілі кластерів: Траєкторії (KMeans) ---")
kmeans_traj = joblib.load(os.path.join(models_dir, 'kmeans_traj_model.pkl'))
centers_scaled_24d = kmeans_traj.cluster_centers_  # shape: (7, 24)
# Reshape to (7 * 4, 6)
centers_scaled_reshaped = centers_scaled_24d.reshape(-1, len(features))
centers_raw_reshaped = np.expm1(scaler.inverse_transform(centers_scaled_reshaped))
# Average across the 4 years
centers_raw_avg = centers_raw_reshaped.reshape(-1, len(years), len(features)).mean(axis=1)

df_tk = pd.DataFrame(centers_raw_avg, columns=features).round(0)
df_tk.index.name = 'Кластер'
df_tk['Авто-Опис'] = df_tk.apply(generate_description, axis=1)
print(df_tk.to_string())
df_tk.to_csv(os.path.join(data_dir, 'cluster_meanings_traj_kmeans.csv'))

# 4. Trajectories Agglomerative (From Empirical Means)
print("\n--- Профілі кластерів: Траєкторії (Математичний DTW) ---")
df_ta_res = df_res[df_res['Вид кластеризації'] == 'Траєкторії (Математичний DTW)'].copy()
df_ta_res['Код громади'] = df_ta_res['Код громади'].astype(str)

df_ta_merged = pd.merge(df_all, df_ta_res, left_on='Айді громади', right_on='Код громади', how='inner')
df_hromada_avg = df_ta_merged.groupby(['Кластер', 'Айді громади'])[features].mean().reset_index()

profile_ta = df_hromada_avg.groupby('Кластер')[features].mean().round(0)
profile_ta['Кількість громад'] = df_hromada_avg.groupby('Кластер').size()
profile_ta['Авто-Опис'] = profile_ta.apply(generate_description, axis=1)
print(profile_ta.to_string())
profile_ta.to_csv(os.path.join(data_dir, 'cluster_meanings_traj_agg.csv'))

print("\nГотово! Результати збережено у папці data.")
