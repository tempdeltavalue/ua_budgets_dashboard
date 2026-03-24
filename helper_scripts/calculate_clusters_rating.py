import pandas as pd
import numpy as np
import os
import json

DATA_DIR = 'pca_t_data'
N_COMP = 3 
LEVEL = 'l3' 

INCOME_WEIGHTS = {
    '11010400': 1.0,  
    '18050400': 1.0,  
    '18050500': 1.0,  
    '18010600': 0.8,  
    '18010900': 0.8,  
    '21081100': 0.5,  
    '41020100': -1.0, 
    '41033900': -0.5  
}

PROG_WEIGHTS = {
    '1184': 1.0,  
    '7413': 0.5,  
    '1021': 0.0,  
    '1091': 0.0,  
    '2010': 0.0,  
    '0160': 0.0,  
    '8600': -0.5, 
    '7692': -1.0, 
    '9220': -1.0 
}

EXPERT_WEIGHTS = {
    'income': INCOME_WEIGHTS,
    'prog': PROG_WEIGHTS
}

def get_zone(score):
    if score >= 30: return "ЗЕЛЕНА (Автономні / Інвестори)"
    elif score >= -10: return "ЖОВТА (Транзитні / Стагнація)"
    else: return "ЧЕРВОНА (Банкрути / Реципієнти)"

print(f"\n{'='*80}")
print(f"РОЗРАХУНОК ФІНАНСОВОГО ЗДОРОВ'Я ТА ТОП-ГРОМАД (Рівень: {LEVEL.upper()})")
print(f"{'='*80}")

comp_dir = os.path.join(DATA_DIR, f"{N_COMP}_comp")

for cat_key in ['income', 'prog']:
    traj_file = os.path.join(comp_dir, "trajectories", f"traj_{cat_key}.json")
    clus_file = os.path.join(comp_dir, "clusters", f"clusters_{cat_key}.json")
    
    if not os.path.exists(traj_file) or not os.path.exists(clus_file):
        continue
        
    with open(traj_file, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)
    with open(clus_file, 'r', encoding='utf-8') as f:
        clus_data = json.load(f)

    metadata = traj_data.get('metadata', {}).get(LEVEL, {})
    pc_scores = {}
    weights_dict = EXPERT_WEIGHTS[cat_key]
    
    for pc_name, features in metadata.items():
        score = 0
        for f in features:
            code = f['code']
            weight = f['weight']
            expert_val = weights_dict.get(code, 0)
            score += weight * expert_val
        pc_scores[pc_name] = score

    trajectories = traj_data.get('trajectories', {})
    cluster_records = []
    
    latest_year = sorted(clus_data.get(LEVEL, {}).keys())[-1]
    current_clusters = clus_data[LEVEL][latest_year]
    
    members_data = []
    
    for code, info in trajectories.items():
        if info['level'] != LEVEL: continue
        
        coords = {}
        for pc in pc_scores.keys():
            coords[pc] = np.mean(info[pc.lower()])
            
        c_info = current_clusters.get(code)
        if c_info:
            cluster_id = c_info.get('cluster')
            distance = c_info.get('distance')
            
            coords['cluster'] = cluster_id
            cluster_records.append(coords)
            
            members_data.append({
                'cluster': cluster_id,
                'name': info.get('name', f'Громада {code}'),
                'distance': distance
            })
            
    df = pd.DataFrame(cluster_records)
    members_df = pd.DataFrame(members_data)
    
    if df.empty: continue
        
    centroids = df.groupby('cluster').mean()
    
    cluster_ratings = []
    for cluster_id, centroid in centroids.iterrows():
        raw_score = sum(centroid[pc] * pc_scores[pc] for pc in pc_scores.keys())
        cluster_ratings.append({'Cluster': cluster_id, 'Raw_Score': raw_score})
        
    ratings_df = pd.DataFrame(cluster_ratings)
    
    max_abs = ratings_df['Raw_Score'].abs().max()
    if max_abs > 0:
        ratings_df['Score_100'] = (ratings_df['Raw_Score'] / max_abs) * 100
    else:
        ratings_df['Score_100'] = 0
        
    ratings_df = ratings_df.sort_values('Score_100', ascending=False)
    
    print(f"\nКАТЕГОРІЯ: {cat_key.upper()} (Рік: {latest_year})")
    print("-" * 80)
    print(f"{'Кл.':<5} | {'Рейтинг':<8} | {'Діагноз / Зона'}")
    print("-" * 80)
    
    for _, row in ratings_df.iterrows():
        c_id = int(row['Cluster'])
        score = round(row['Score_100'], 1)
        zone = get_zone(score)
        
        print(f"{c_id:<5} | {score:>8} | {zone}")
        
        if not members_df.empty:
            c_members = members_df[members_df['cluster'] == c_id]
            top_3 = c_members.sort_values('distance').head(3)
            names_list = [f"{r['name']} ({r['distance']:.2f})" for _, r in top_3.iterrows()]
            
            if names_list:
                print(f"       Еталони: {', '.join(names_list)}")
        print("-" * 80)

print(f"\n{'='*80}")