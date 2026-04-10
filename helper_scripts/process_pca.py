import pandas as pd
import numpy as np
import os
import re
import json
import time
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

BASE_DIR = 'b_data'
OUTPUT_BASE_DIR = 'pca_t_data'

for n_comp in [3, 10]:
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, f"{n_comp}_comp", "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, f"{n_comp}_comp", "errors"), exist_ok=True)

REGION_MAP = {
    '02': 'Vinnytska', '03': 'Volynska', '04': 'Dnipropetrovska',
    '05': 'Donetska', '06': 'Zhytomyrska', '07': 'Zakarpatska',
    '08': 'Zaporizka', '09': 'Ivano-Frankivska', '10': 'Kyivska',
    '11': 'Kirovohradska', '12': 'Luhanska', '13': 'Lvivska',
    '14': 'Mykolaivska', '15': 'Odeska', '16': 'Poltavska',
    '17': 'Rivnenska', '18': 'Sumska', '19': 'Ternopilska',
    '20': 'Kharkivska', '21': 'Khersonska', '22': 'Khmelnytska',
    '23': 'Cherkaska', '24': 'Chernivetska', '25': 'Chernihivska',
    '26': 'Kyiv City'
}

CATEGORIES = {
    'income': {'folder': 'empty_incomes', 'suffix': 'income', 'label': 'REVENUES (Own receipts)'},
    'prog': {'folder': 'program_expenses', 'suffix': 'prog', 'label': 'EXPENSES (Programmatic)'},
    'econ': {'folder': 'economic_expenses', 'suffix': 'econ', 'label': 'EXPENSES (Economic)'},
    'func': {'folder': 'functional_expenses', 'suffix': 'func', 'label': 'EXPENSES (Functional)'}
}

def process_single_file(task):
    file_path, name_part, code, root_path, cat_key = task
    try:
        lvl = 'l1' if '1_' in root_path else 'l2' if '2_' in root_path else 'l3' if '3_' in root_path else 'unknown'
        if lvl == 'unknown': return None
        
        df_i = pd.read_csv(file_path, sep=';', encoding='utf-8-sig', dtype=str).dropna(how='all')
        cols = [c.upper() for c in df_i.columns]
        
        if 'FUND_TYP' in cols and cat_key != 'income': 
            df_i = df_i[df_i[df_i.columns[cols.index('FUND_TYP')]].str.upper().isin(['C', 'T', 'З', 'С', 'S'])]
            if df_i.empty: return None
            
        p_col = next((c for c in cols if 'REP_PERIOD' in c), None)
        c_col_candidates = [c for c in cols if ('COD' in c or 'PROG' in c or 'EK' in c or 'FK' in c or 'INC' in c) and c != 'COD_BUDGET']
        if not c_col_candidates or not p_col: return None
        k_col = c_col_candidates[0]
        
        best_sum = -1
        v_col = None
        best_series = None
        
        for ac in [c for c in cols if any(k in c for k in ['FAKT', 'EXEC', 'ZAT', 'PLANS', 'AMT', 'SUM'])]:
            clean_series = pd.to_numeric(df_i[ac].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '', regex=False), errors='coerce').fillna(0)
            temp_sum = clean_series.sum()
            if temp_sum > best_sum: 
                best_sum = temp_sum
                v_col = ac
                best_series = clean_series
                
        if not v_col or best_sum <= 0: return None
        
        temp = pd.DataFrame({'REP_PERIOD': df_i[p_col], 'COD': df_i[k_col], 'AMT': best_series})
        temp = temp[temp['AMT'] > 0]
        if temp.empty: return None
        
        temp['COD'] = temp['COD'].astype(str).str.replace('.0', '', regex=False).str.strip()
        if cat_key == 'income': temp['COD'] = temp['COD'].str.zfill(8).str[-8:]
        elif cat_key == 'prog': temp['COD'] = temp['COD'].str.zfill(7).str[-4:]
        else: temp['COD'] = temp['COD'].str.zfill(4).str[-4:]
            
        temp['REP_PERIOD'] = temp['REP_PERIOD'].astype(str).str.strip()
        temp['BUDGET_CODE'] = code
        temp['LEVEL'] = lvl
        
        agg_temp = temp.groupby(['BUDGET_CODE', 'REP_PERIOD', 'LEVEL', 'COD'], as_index=False)['AMT'].sum()
        
        n_col = next((c for c in cols if 'NAME' in c or 'НАЗВ' in c), None)
        local_names = {}
        if n_col:
            names_df = df_i[[k_col, n_col]].dropna().drop_duplicates(subset=[k_col]).copy()
            names_df[k_col] = names_df[k_col].astype(str).str.replace('.0', '', regex=False).str.strip()
            if cat_key == 'income': names_df[k_col] = names_df[k_col].str.zfill(8).str[-8:]
            elif cat_key == 'prog': names_df[k_col] = names_df[k_col].str.zfill(7).str[-4:]
            else: names_df[k_col] = names_df[k_col].str.zfill(4).str[-4:]
                
            names_df[n_col] = names_df[n_col].astype(str).str.split(';;').str[0].str.strip()
            local_names = dict(zip(names_df[k_col], names_df[n_col]))
            
        return (agg_temp, local_names, code, name_part)
    except Exception:
        return None

if __name__ == '__main__':
    global_start_time = time.time()
    print("-" * 70)
    print("STARTING GLOBAL CALCULATION")
    print("-" * 70)

    for cat_key, cat_info in CATEGORIES.items():
        cat_start_time = time.time()
        
        folder_name = cat_info['folder']
        file_suffix = cat_info['suffix']
        label = cat_info['label']

        print("-" * 50)
        print(f"PROCESSING CATEGORY: {label}")
        print("-" * 50)

        print(f"[1/4] Reading files from '{folder_name}'...")
        tasks = []
        for root, dirs, files in os.walk(BASE_DIR):
            if folder_name in root:
                match = re.search(r'(.*)_(\d{10})', os.path.basename(os.path.dirname(root)))
                if match and match.group(2)[:2] in REGION_MAP:
                    for f in glob.glob(os.path.join(root, '*.csv')):
                        tasks.append((f, match.group(1), match.group(2), root, cat_key))

        total_files = len(tasks)
        if total_files == 0:
            print(f"No files found for {cat_key}.")
            continue

        all_records = []
        GLOBAL_CODE_NAMES = {}
        budget_names = {}
        
        processed = 0
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_single_file, task): task for task in tasks}
            for future in as_completed(futures):
                processed += 1
                if processed % 200 == 0:
                    print(f"   ... processed {processed}/{total_files} files")
                    
                res = future.result()
                if res is not None:
                    agg_temp, local_names, code, name_part = res
                    all_records.append(agg_temp)
                    GLOBAL_CODE_NAMES.update(local_names)
                    budget_names[code] = name_part

        if not all_records:
            continue

        print("Aggregating data...")
        full_df = pd.concat(all_records, ignore_index=True)
        full_df = full_df.groupby(['BUDGET_CODE', 'REP_PERIOD', 'LEVEL', 'COD'], as_index=False)['AMT'].sum()
        full_df['Date'] = pd.to_datetime(full_df['REP_PERIOD'], format='%m.%Y', errors='coerce')
        full_df = full_df.dropna(subset=['Date', 'BUDGET_CODE'])

        for n_comp in [3, 10]:
            print(f"[3/4] Calculating PCA and reconstruction errors ({n_comp} components)...")
            
            pca_results = {}
            pca_metadata = {}
            error_results = {}

            for current_level in ['l1', 'l2', 'l3']:
                df_level = full_df[full_df['LEVEL'] == current_level]
                if df_level.empty: continue
                    
                pivot = df_level.pivot_table(index=['BUDGET_CODE', 'Date'], columns='COD', values='AMT', aggfunc='sum', fill_value=0)
                total_monthly = pivot.sum(axis=1)
                pivot_percent = pivot.div(total_monthly.replace(0, np.nan), axis=0).fillna(0)

                smoothed_data = pivot_percent.groupby(level='BUDGET_CODE', group_keys=False).apply(
                    lambda x: x.rolling(window=12, min_periods=3).mean()
                ).dropna()

                smoothed_size = total_monthly.groupby(level='BUDGET_CODE', group_keys=False).apply(
                    lambda x: x.rolling(window=12, min_periods=3).mean()
                ).loc[smoothed_data.index]

                if smoothed_data.empty: continue

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(smoothed_data)
                
                actual_comp = min(n_comp, X_scaled.shape[1], X_scaled.shape[0])
                pca = PCA(n_components=actual_comp)
                
                components = pca.fit_transform(X_scaled)
                X_reconstructed = pca.inverse_transform(components)
                reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

                features = smoothed_data.columns
                loadings = pca.components_
                level_metadata = {}
                pc_names = [f'PC{i+1}' for i in range(actual_comp)]
                
                for i, pc in enumerate(pc_names):
                    weights_df = pd.DataFrame({'code': features, 'weight': loadings[i]})
                    top_features = weights_df.reindex(weights_df['weight'].abs().sort_values(ascending=False).index).head(5)
                    feature_list = [{"code": row['code'], "name": GLOBAL_CODE_NAMES.get(row['code'], f"Item {row['code']}"), "weight": round(row['weight'], 4), "direction": "positive" if row['weight'] > 0 else "negative"} for _, row in top_features.iterrows()]
                    level_metadata[pc] = feature_list
                pca_metadata[current_level] = level_metadata

                pca_df = pd.DataFrame(data=components, columns=pc_names, index=smoothed_data.index).reset_index()
                pca_df['Budget_Size'] = smoothed_size.values
                pca_df['Date_Str'] = pca_df['Date'].dt.strftime('%m.%Y')
                pca_df['Recon_Error'] = reconstruction_errors

                error_results[current_level] = {}

                for code, group in pca_df.groupby('BUDGET_CODE'):
                    group = group.sort_values('Date')
                    
                    local_top_factors = []
                    for _, row in group.iterrows():
                        pt = smoothed_data.loc[(code, row['Date'])].nlargest(5)
                        f_data = [{"name": GLOBAL_CODE_NAMES.get(c, c)[:50], "share": round(v * 100, 2)} for c, v in pt.items() if v > 0.001]
                        local_top_factors.append(f_data if f_data else [{"name": "Other", "share": 100.0}])

                    avg_raw = smoothed_data.loc[code].mean().nlargest(5)
                    overall_top_5 = [{"name": GLOBAL_CODE_NAMES.get(c, c)[:50], "share": round(v * 100, 2)} for c, v in avg_raw.items() if v > 0.001]
                    
                    trajectory_data = {
                        "name": budget_names.get(code, "Unknown"), "level": current_level, "dates": group['Date_Str'].tolist(),
                        "size_mln": (group['Budget_Size'] / 1e6).round(2).tolist(), "overall_top_5": overall_top_5, "top_factors": local_top_factors
                    }
                    for pc in pc_names:
                        trajectory_data[pc.lower()] = group[pc].round(4).tolist()
                    
                    pca_results[code] = trajectory_data
                    
                    error_results[current_level][code] = {
                        "name": budget_names.get(code, "Unknown"),
                        "dates": group['Date_Str'].tolist(),
                        "error": group['Recon_Error'].round(4).tolist()
                    }

            print(f"[4/4] Saving output files for {n_comp} components...")
            
            traj_path = os.path.join(OUTPUT_BASE_DIR, f"{n_comp}_comp", "trajectories", f"traj_{file_suffix}.json")
            err_path = os.path.join(OUTPUT_BASE_DIR, f"{n_comp}_comp", "errors", f"error_{file_suffix}.json")
            
            with open(traj_path, 'w', encoding='utf-8') as f:
                json.dump({"metadata": pca_metadata, "trajectories": pca_results}, f, ensure_ascii=False)
                
            with open(err_path, 'w', encoding='utf-8') as f:
                json.dump(error_results, f, ensure_ascii=False)

        print(f"Category '{cat_key}' processed in {(time.time() - cat_start_time)/60:.1f} min.")

    print("-" * 70)
    print(f"GLOBAL CYCLE COMPLETED IN {(time.time() - global_start_time)/60:.1f} min.")