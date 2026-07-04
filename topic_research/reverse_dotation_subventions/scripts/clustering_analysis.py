import os
import pandas as pd
import geopandas as gpd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cdist
import warnings
warnings.filterwarnings("ignore")

data_dir = "article_data/data"
models_dir = "article_data/models"
os.makedirs(models_dir, exist_ok=True)

years = [2022, 2023, 2024, 2025]
features = [
    'Сума Базова', 'Сума Реверсна', 'Сума Додаткові',
    'Сума Війна та відновлення', 'Сума Освіта', "Сума Охорона здоров'я"
]

print("Loading master hromada list from l3.geojson...")
import json
with open('t_app/viz_dashboard_data2/l3.geojson', 'r', encoding='utf-8') as f:
    geojson = json.load(f)
master_hromadas = pd.DataFrame([f['properties'] for f in geojson['features']])
master_hromadas = master_hromadas.rename(columns={'BUDGET_CODE': 'Айді громади', 'display_name': 'Назва громади'})
master_hromadas = master_hromadas[['Айді громади', 'Назва громади']]
master_hromadas['Айді громади'] = master_hromadas['Айді громади'].astype(str).str.strip()

print("Loading data...")
merged_data = {}
hromada_names = dict(zip(master_hromadas['Айді громади'], master_hromadas['Назва громади']))

for year in years:
    dot_file = os.path.join(data_dir, f'dotations_{year}.csv')
    sub_file = os.path.join(data_dir, f'subventions_{year}.csv')
    
    df_base = master_hromadas.copy()
    
    if os.path.exists(dot_file) and os.path.exists(sub_file):
        df_dot = pd.read_csv(dot_file, dtype={'Айді громади': str})
        df_sub = pd.read_csv(sub_file, dtype={'Айді громади': str})
        
        df_dot['Айді громади'] = df_dot['Айді громади'].astype(str).str.strip()
        df_sub['Айді громади'] = df_sub['Айді громади'].astype(str).str.strip()
        
        # Left join onto master list
        df_merged = pd.merge(df_base, df_dot, on='Айді громади', how='left')
        df_merged = pd.merge(df_merged, df_sub, on='Айді громади', how='left', suffixes=('', '_sub'))
        
        X = df_merged[['Айді громади'] + features].copy()
        X[features] = X[features].fillna(0.0)
        merged_data[year] = X
    else:
        # Fallback to zeros if files are missing
        df_base[features] = 0.0
        merged_data[year] = df_base[['Айді громади'] + features]

print("Preparing Pooled Data...")
pooled_rows = []
for year in years:
    df_year = merged_data[year]
    for _, row in df_year.iterrows():
        r = row.to_dict()
        r['Рік'] = year
        pooled_rows.append(r)
        
df_pooled = pd.DataFrame(pooled_rows)
X_pooled = df_pooled[features].values

# We need to scale the data robustly to prevent outliers from dominating.
# Log1p + StandardScaler often works better for highly skewed financial data than QuantileTransformer
scaler = StandardScaler()
X_pooled_log = np.log1p(np.maximum(X_pooled, 0))
X_pooled_scaled = scaler.fit_transform(X_pooled_log)
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

# Store scaled features back into the dataframe for easy DTW extraction
for i, col in enumerate(features):
    df_pooled[f"{col}_scaled"] = X_pooled_scaled[:, i]

print("1. Running Pooled KMeans (n=5)...")
kmeans_pooled = KMeans(n_clusters=5, random_state=42, n_init=10)
pooled_kmeans_labels = kmeans_pooled.fit_predict(X_pooled_scaled)
joblib.dump(kmeans_pooled, os.path.join(models_dir, 'kmeans_pooled_model.pkl'))

pooled_kmeans_distances = np.zeros(len(X_pooled_scaled))
for i in range(5):
    mask = (pooled_kmeans_labels == i)
    if np.sum(mask) > 0:
        centroid = kmeans_pooled.cluster_centers_[i]
        dists = cdist(X_pooled_scaled[mask], [centroid], metric='euclidean').flatten()
        pooled_kmeans_distances[mask] = dists

print("2. Running Pooled Agglomerative (n=5)...")
agg_pooled = AgglomerativeClustering(n_clusters=5)
pooled_agg_labels = agg_pooled.fit_predict(X_pooled_scaled)
joblib.dump(agg_pooled, os.path.join(models_dir, 'agg_pooled_model.pkl'))

pooled_agg_distances = np.zeros(len(X_pooled_scaled))
unique_pooled_agg_labels = np.unique(pooled_agg_labels)
pooled_agg_medoids = {}
for label in unique_pooled_agg_labels:
    indices = np.where(pooled_agg_labels == label)[0]
    sub_X = X_pooled_scaled[indices]
    dist_mat = cdist(sub_X, sub_X)
    medoid_idx = np.argmin(dist_mat.sum(axis=1))
    pooled_agg_medoids[label] = sub_X[medoid_idx]

for i, label in enumerate(pooled_agg_labels):
    pooled_agg_distances[i] = euclidean(X_pooled_scaled[i], pooled_agg_medoids[label])

# 3. Trajectory Clustering (DTW)
print("Preparing Trajectories Data...")
trajectories = {}
unique_hromadas = list(hromada_names.keys())
N = len(unique_hromadas)

scaled_features = [f"{col}_scaled" for col in features]

X_traj_flattened = []
for code in unique_hromadas:
    traj = []
    for year in years:
        row = df_pooled[(df_pooled['Айді громади'] == code) & (df_pooled['Рік'] == year)]
        if not row.empty:
            vec = row[scaled_features].values[0]
            traj.append(vec)
        else:
            traj.append(np.zeros(len(features)))
    traj_arr = np.array(traj)
    trajectories[code] = traj_arr
    X_traj_flattened.append(traj_arr.flatten())

X_traj_flattened = np.array(X_traj_flattened)

print("3. Running Trajectories KMeans (n=7)...")
kmeans_traj = KMeans(n_clusters=7, random_state=42, n_init=10)
traj_kmeans_labels = kmeans_traj.fit_predict(X_traj_flattened)
joblib.dump(kmeans_traj, os.path.join(models_dir, 'kmeans_traj_model.pkl'))

traj_kmeans_distances = np.zeros(len(X_traj_flattened))
for i in range(7):
    mask = (traj_kmeans_labels == i)
    if np.sum(mask) > 0:
        centroid = kmeans_traj.cluster_centers_[i]
        dists = cdist(X_traj_flattened[mask], [centroid], metric='euclidean').flatten()
        traj_kmeans_distances[mask] = dists

dtw_matrix_path = os.path.join(models_dir, 'dtw_distance_matrix.npy')
if os.path.exists(dtw_matrix_path):
    print("Loading precomputed DTW distance matrix...")
    distance_matrix = np.load(dtw_matrix_path)
else:
    print(f"Computing DTW distance matrix for {N} hromadas (this might take a minute)...")
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        if i % 100 == 0:
            print(f"Processed {i}/{N} trajectories...")
        seq_i = trajectories[unique_hromadas[i]]
        for j in range(i+1, N):
            seq_j = trajectories[unique_hromadas[j]]
            dist, _ = fastdtw(seq_i, seq_j, dist=euclidean)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    np.save(dtw_matrix_path, distance_matrix)

print("4. Running Trajectories Agglomerative on DTW (n=7)...")
agg_dtw = AgglomerativeClustering(n_clusters=7, metric='precomputed', linkage='average')
dtw_agg_labels = agg_dtw.fit_predict(distance_matrix)
joblib.dump(agg_dtw, os.path.join(models_dir, 'agg_dtw_model.pkl'))

dtw_agg_distances = np.zeros(N)
unique_dtw_agg_labels = np.unique(dtw_agg_labels)
dtw_medoids = {}
for label in unique_dtw_agg_labels:
    indices = np.where(dtw_agg_labels == label)[0]
    sub_matrix = distance_matrix[np.ix_(indices, indices)]
    medoid_idx_in_sub = np.argmin(sub_matrix.sum(axis=1))
    dtw_medoids[label] = indices[medoid_idx_in_sub]

for i, label in enumerate(dtw_agg_labels):
    dtw_agg_distances[i] = distance_matrix[i, dtw_medoids[label]]

# Save results to CSV
print("Formatting output CSV...")
results = []

for idx in range(len(df_pooled)):
    code = df_pooled.iloc[idx]['Айді громади']
    yr = df_pooled.iloc[idx]['Рік']
    name = hromada_names.get(code, 'Unknown')
    
    results.append({
        'Код громади': code, 'Назва громади': name,
        'Вид кластеризації': 'Пул (KMeans)', 'Рік': yr,
        'Кластер': pooled_kmeans_labels[idx], 'Відстань до центроїда': pooled_kmeans_distances[idx]
    })
    results.append({
        'Код громади': code, 'Назва громади': name,
        'Вид кластеризації': 'Пул (Математичний)', 'Рік': yr,
        'Кластер': pooled_agg_labels[idx], 'Відстань до центроїда': pooled_agg_distances[idx]
    })

for i, code in enumerate(unique_hromadas):
    name = hromada_names.get(code, 'Unknown')
    
    results.append({
        'Код громади': code, 'Назва громади': name,
        'Вид кластеризації': 'Траєкторії (KMeans)', 'Рік': '2022-2025',
        'Кластер': traj_kmeans_labels[i], 'Відстань до центроїда': traj_kmeans_distances[i]
    })
    results.append({
        'Код громади': code, 'Назва громади': name,
        'Вид кластеризації': 'Траєкторії (Математичний DTW)', 'Рік': '2022-2025',
        'Кластер': dtw_agg_labels[i], 'Відстань до центроїда': dtw_agg_distances[i]
    })

df_results = pd.DataFrame(results)
output_path = os.path.join(data_dir, 'clustering_results.csv')
df_results.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Done! Results saved to {output_path}")
print(f"Models saved in {models_dir}/")
