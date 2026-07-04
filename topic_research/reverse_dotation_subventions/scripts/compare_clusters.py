import pandas as pd

# Завантажуємо результати всіх 4-х алгоритмів
df = pd.read_csv('article_data/data/clustering_results.csv')

# --- 1. Порівняння Траєкторій (KMeans проти Математичного DTW) ---
df_traj_k = df[df['Вид кластеризації'] == 'Траєкторії (KMeans)'].rename(columns={'Кластер': 'KMeans'})
df_traj_a = df[df['Вид кластеризації'] == 'Траєкторії (Математичний DTW)'].rename(columns={'Кластер': 'DTW'})

# Зводимо в одну таблицю по Коду громади
merged_traj = pd.merge(df_traj_k[['Код громади', 'KMeans']], df_traj_a[['Код громади', 'DTW']], on='Код громади')

print('\n--- Перетин кластерів: Траєкторії (KMeans по вертикалі, DTW по горизонталі) ---')
print(pd.crosstab(merged_traj['KMeans'], merged_traj['DTW']))


# --- 2. Порівняння Об'єднаного пулу (KMeans проти Математичного Agglomerative) ---
df_pool_k = df[df['Вид кластеризації'] == 'Пул (KMeans)'].rename(columns={'Кластер': 'KMeans'})
df_pool_a = df[df['Вид кластеризації'] == 'Пул (Математичний)'].rename(columns={'Кластер': 'Agglomerative'})

# Зводимо в одну таблицю по Коду громади та Року
merged_pool = pd.merge(df_pool_k[['Код громади', 'Рік', 'KMeans']], df_pool_a[['Код громади', 'Рік', 'Agglomerative']], on=['Код громади', 'Рік'])

print('\n--- Перетин кластерів: Пул (KMeans по вертикалі, Agglomerative по горизонталі) ---')
print(pd.crosstab(merged_pool['KMeans'], merged_pool['Agglomerative']))
