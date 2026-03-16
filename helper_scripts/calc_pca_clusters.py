import pandas as pd
import numpy as np
import os
import json
import time
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

start_time = time.time()

# --- 1. НАЛАШТУВАННЯ ---
DATA_DIR = 'pca_t_data'

# Словник категорій: {назва_для_виводу: суфікс_файлу}
# Назви файлів тепер уніфіковані: traj_income.json, traj_prog.json і т.д.
CATEGORIES = {
    'income': 'income',
    'prog': 'prog',
    'econ': 'econ',
    'func': 'func'
}

# Кількість кластерів за рівнями
CLUSTERS_CONFIG = {
    'l1': 3,   # Області
    'l2': 5,   # Райони
    'l3': 10   # Громади
}

print(f"\n{'='*60}")
print(f"🚀 Починаю ГЛОБАЛЬНУ кластеризацію для різних розмірностей")
print(f"{'='*60}")

# Проходимо по обох розмірностях (3 та 10 компонент)
for n_comp in [3, 10]:
    comp_dir = os.path.join(DATA_DIR, f"{n_comp}_comp")
    traj_dir = os.path.join(comp_dir, "trajectories")
    
    # Створюємо нову папку clusters
    clusters_dir = os.path.join(comp_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)
    
    print(f"\n📂 Обробка простору: {n_comp} компонент")
    
    for cat_key, suffix in CATEGORIES.items():
        input_file = os.path.join(traj_dir, f"traj_{suffix}.json")
        output_file = os.path.join(clusters_dir, f"clusters_{suffix}.json")

        print(f"\n   🔄 Категорія: {cat_key.upper()}")
        
        # Перевірка наявності файлу PCA
        if not os.path.exists(input_file):
            print(f"   ⚠️ Файл {input_file} не знайдено! Пропускаю...")
            continue

        print("      📥 Завантажую дані PCA...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trajectories = data.get('trajectories', {})

        if not trajectories:
            print("      ❌ Немає даних траєкторій у файлі. Пропускаю...")
            continue

        print("      🧹 Підготовка даних (усереднення точок по роках)...")
        rows = []
        for code, info in trajectories.items():
            lvl = info['level']
            
            # Динамічно шукаємо всі наявні компоненти (pc1, pc2... pc10)
            pc_keys = [k for k in info.keys() if k.startswith('pc') and k[2:].isdigit()]
            
            for i, date_str in enumerate(info['dates']):
                year = date_str.split('.')[1]
                row_data = {
                    'code': code,
                    'level': lvl,
                    'year': year
                }
                # Додаємо дані всіх знайдених компонент
                for pc in pc_keys:
                    row_data[pc] = info[pc][i]
                    
                rows.append(row_data)

        df = pd.DataFrame(rows)
        pc_columns = [c for c in df.columns if c.startswith('pc')]

        # Знаходимо середнє положення кожної громади/області в просторі для кожного року
        annual_df = df.groupby(['level', 'year', 'code'])[pc_columns].mean().reset_index()

        print(f"      🤖 Запуск KMeans (використовується {len(pc_columns)} осей)...")
        results = {'l1': {}, 'l2': {}, 'l3': {}}

        for lvl in ['l1', 'l2', 'l3']:
            lvl_df = annual_df[annual_df['level'] == lvl].copy()
            if lvl_df.empty:
                continue
                
            unique_codes = len(lvl_df['code'].unique())
            target_k = min(CLUSTERS_CONFIG[lvl], unique_codes)
            
            if target_k < 1:
                continue
                
            # ТРЕНУЄМО МОДЕЛЬ
            X_all = lvl_df[pc_columns].values
            kmeans = KMeans(n_clusters=target_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_all)
            
            # Розраховуємо відстані
            distances_to_all = kmeans.transform(X_all)
            assigned_distances = [distances_to_all[i, label] for i, label in enumerate(labels)]
            
            lvl_df['cluster'] = labels + 1
            lvl_df['distance'] = assigned_distances
            
            # ФОРМУЄМО РЕЗУЛЬТАТ ПО РОКАХ
            years = lvl_df['year'].unique()
            for year in sorted(years):
                results[lvl][year] = {}
                year_df = lvl_df[lvl_df['year'] == year]
                
                for _, row in year_df.iterrows():
                    results[lvl][year][row['code']] = {
                        'cluster': int(row['cluster']),
                        'distance': round(float(row['distance']), 4)
                    }

        print(f"      💾 Зберігаю результати: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"🎉 ГОТОВО! Загальний час виконання: {time.time() - start_time:.2f} сек.")