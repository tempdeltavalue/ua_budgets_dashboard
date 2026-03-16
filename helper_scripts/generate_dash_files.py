import pandas as pd
import numpy as np
import glob
import os
import geopandas as gpd
import re
import warnings
import json
import difflib
import time

warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

start_time = time.time()

# --- 1. НАЛАШТУВАННЯ ---
BASE_DIR = 'parq_b_data' # Змінили папку
CACHE_DIR = "viz_dashboard_data_parq" # Змінили кеш, щоб скрипт точно все перерахував

GEO_PATH = "ukr_admin_boundaries/ukr_admin3.geojson" 
OUTPUT_HTML = "all_ukraine_ultimate.html" # Змінив назву
YEARS = list(range(2018, 2026)) 

# Словник для всієї України
ALL_OBLASTS = {
    '01': 'Crimea', # АБО точна назва з твого geojson файлу
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

# --- ПЕРЕВІРКА КЕШУ ---
files_needed = ['l1.geojson', 'l2.geojson', 'l3.geojson', 'ui_config.json']
cache_exists = all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in files_needed)

if cache_exists:
    print(f"⚡ ЗНАЙДЕНО КЕШ у '{CACHE_DIR}'! Моментальна генерація...")
    with open(os.path.join(CACHE_DIR, 'l1.geojson'), 'r', encoding='utf-8') as f: l1_json = f.read()
    with open(os.path.join(CACHE_DIR, 'l2.geojson'), 'r', encoding='utf-8') as f: l2_json = f.read()
    with open(os.path.join(CACHE_DIR, 'l3.geojson'), 'r', encoding='utf-8') as f: l3_json = f.read()
    with open(os.path.join(CACHE_DIR, 'ui_config.json'), 'r', encoding='utf-8') as f: ui_json = f.read()

else:
    print(f"🐌 Кеш не знайдено. Починаю повний цикл розрахунків...")
    
    BUDGET_STATS = {y: {} for y in YEARS}
    BUDGET_NAMES = {}
    CODE_LEVELS = {} 
    GLOBAL_CODE_NAMES = {cat: {} for cat in ['INC', 'PROG', 'ECON', 'FUNC']}

    # --- 2. ПАРСИНГ ---
    print("📊 [1/4] Читаю фінансові дані...")
    for root, dirs, files in os.walk(BASE_DIR):
        match = re.search(r'(.*)_(\d{10})', os.path.basename(root))
        if not match: continue
        name_part, code = match.group(1), match.group(2)
        
        # ВИКОРИСТОВУЄМО ALL_OBLASTS
        if code[:2] not in ALL_OBLASTS: continue 
        
        lvl = 'unknown'
        if '1_' in root: lvl = 'l1'
        elif '2_' in root: lvl = 'l2'
        elif '3_' in root: lvl = 'l3'
        
        if lvl == 'unknown': continue 
            
        CODE_LEVELS[code] = lvl
        BUDGET_NAMES[code] = name_part
        
        for y in YEARS:
            if code not in BUDGET_STATS[y]: BUDGET_STATS[y][code] = {'INC': {}, 'PROG': {}, 'ECON': {}, 'FUNC': {}}
            for folder, cat_key in {'empty_incomes': 'INC', 'program_expenses': 'PROG', 'economic_expenses': 'ECON', 'functional_expenses': 'FUNC'}.items():
                for f in glob.glob(os.path.join(root, folder, f'*{y}*.parquet')): # <-- ЗМІНЕНО ТУТ
                    try:
                        # Читаємо Parquet! Швидко і без костилів з кодуваннями
                        df = pd.read_parquet(f).dropna(how='all') # <-- ЗМІНЕНО ТУТ
                        
                        cols = [c.upper() for c in df.columns]
                        if 'FUND_TYP' in cols and cat_key != 'INC': 
                            df = df[df[df.columns[cols.index('FUND_TYP')]].astype(str).str.upper().isin(['C', 'T', 'З', 'С', 'S'])].copy()
                        
                        c_col_candidates = [c for c in cols if ('COD' in c or 'PROG' in c or 'EK' in c or 'FK' in c or 'INC' in c) and c != 'COD_BUDGET']
                        if not c_col_candidates: continue
                        orig_c = df.columns[cols.index(c_col_candidates[0])]
                        
                        best_a, best_sum = None, -1
                        for ac in [c for c in cols if ('FAKT' in c or 'EXEC' in c or 'AMT' in c or 'SUM' in c) and 'PLAN' not in c and 'ZAT' not in c]:                            
                            orig_ac = df.columns[cols.index(ac)]
                            # .astype(str) залишається, щоб захистити код, якщо Parquet зберіг числа як float
                            temp_sum = pd.to_numeric(df[orig_ac].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce').fillna(0).sum()
                            if temp_sum > best_sum: best_sum, best_a = temp_sum, orig_ac
                                
                        if not best_a or best_sum <= 0: continue
                        
                        n_col = next((c for c in cols if 'NAME' in c or 'НАЗВ' in c), None)
                        orig_n = df.columns[cols.index(n_col)] if n_col else None
                        
                        df['C'] = df[orig_c].astype(str).str.replace('.0','',regex=False).str.strip()
                        df['A'] = pd.to_numeric(df[best_a].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce').fillna(0)
                        
                        for _, row in df[df['A'] > 0].iterrows():
                            c_val, amt_val = row['C'], row['A']
                            if cat_key == 'PROG': c_val = str(c_val).zfill(7)[-4:]
                            elif cat_key == 'INC': c_val = str(c_val).zfill(8)[-8:]
                            else: c_val = str(c_val).zfill(4)[-4:]
                            
                            if orig_n:
                                n_val = str(row[orig_n]).split(';;')[0].strip()
                                if len(n_val) > len(GLOBAL_CODE_NAMES[cat_key].get(c_val, '')): GLOBAL_CODE_NAMES[cat_key][c_val] = n_val
                                    
                            BUDGET_STATS[y][code][cat_key][c_val] = max(BUDGET_STATS[y][code][cat_key].get(c_val, 0), amt_val)
                    except Exception as e: 
                        pass # Якщо файл битий, просто йдемо далі

    # --- 3. АГРЕГАЦІЯ ---
    print("🧮 [2/4] Агрегую коди...")
    AGG_STATS = {y: {} for y in YEARS}
    UI_CODES = {y: {l: {cat: {d: {} for d in [1, 2, 4, 8]} for cat in ['INC','PROG','ECON','FUNC']} for l in ['l1','l2','l3']} for y in YEARS}

    for y in YEARS:
        for b_code, cats in BUDGET_STATS[y].items():
            lvl = CODE_LEVELS.get(b_code, 'unknown')
            AGG_STATS[y][b_code] = {'INC': {}, 'PROG': {}, 'ECON': {}, 'FUNC': {}}
            
            for cat in ['INC', 'PROG', 'ECON', 'FUNC']:
                for full_code, amt in cats[cat].items():
                    for d in [1, 2, 4, 8]:
                        if d > len(full_code): continue
                        prefix = full_code[:d]
                        AGG_STATS[y][b_code][cat][prefix] = AGG_STATS[y][b_code][cat].get(prefix, 0) + amt
                        if lvl != 'unknown': UI_CODES[y][lvl][cat][d][prefix] = UI_CODES[y][lvl][cat][d].get(prefix, 0) + amt

            own_inc = AGG_STATS[y][b_code]['INC'].get('1', 0) + AGG_STATS[y][b_code]['INC'].get('2', 0) + AGG_STATS[y][b_code]['INC'].get('3', 0) + AGG_STATS[y][b_code]['INC'].get('5', 0)
            tot_exp = AGG_STATS[y][b_code]['PROG'].get('0000', sum(cats['PROG'].values()))
            AGG_STATS[y][b_code]['TOT_INC'] = own_inc
            AGG_STATS[y][b_code]['TOT_EXP'] = tot_exp
            AGG_STATS[y][b_code]['BAL'] = own_inc - tot_exp

    FINAL_UI = {y: {l: {cat: {d: {} for d in [1, 2, 4, 8]} for cat in ['INC','PROG','ECON','FUNC']} for l in ['l1','l2','l3']} for y in YEARS}
    for y in YEARS:
        for l in ['l1','l2','l3']:
            for cat in ['INC','PROG','ECON','FUNC']:
                for d in [1, 2, 4, 8]:
                    sorted_items = sorted(UI_CODES[y][l][cat][d].items(), key=lambda x: x[1], reverse=True)
                    for code, _ in sorted_items:
                        name = GLOBAL_CODE_NAMES[cat].get(code, f"Група {code}")
                        FINAL_UI[y][l][cat][d][code] = f"{code} - {name[:60]}"

    # --- 4. ГЕО-ОБРОБКА ТА СНАЙПЕРСЬКЕ ЗШИВАННЯ ---
    print("🗺️ [3/4] Зшиваю з картою (Super Match)...")
    gdf = gpd.read_file(GEO_PATH)
    # ВИКОРИСТОВУЄМО ALL_OBLASTS
    allowed_oblasts = list(ALL_OBLASTS.values())
    gdf = gdf[gdf['adm1_name'].isin(allowed_oblasts)].copy()
    gdf['geometry'] = gdf['geometry'].simplify(0.005, preserve_topology=True)

    TRANSLIT_MAP = {
        'shch': 'щ', 'sh': 'ш', 'ch': 'ч', 'zh': 'ж', 'kh': 'х', 'ts': 'ц',
        'tia': 'тя', 'ia': 'я', 'iu': 'ю', 'yu': 'ю', 'ya': 'я', 'ye': 'е', 'yi': 'і', 'ie': 'е',
        'a': 'а', 'b': 'б', 'v': 'в', 'w': 'в', 'h': 'г', 'g': 'г', 'd': 'д', 'e': 'е',
        'z': 'з', 'i': 'і', 'y': 'и', 'j': 'и', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у', 'f': 'ф', 'c': 'к'
    }

    def normalize_and_chop(target):
        target = target.lower()
        for lat, cyr in TRANSLIT_MAP.items(): 
            target = target.replace(lat, cyr)
            
        target = target.replace('ї', 'і').replace('й', 'и').replace('є', 'е').replace('ґ', 'г')
        target = target.replace("'", "").replace("’", "").replace("`", "").replace('ь', '')
        
        target = re.sub(r'[^а-яіиег\-]', '', target)
        
        endings = [
            'ому', 'ою', 'оі', 'ии', 'іи', 'иі', 'ов', 'ого', 'их', 'ів', 
            'а', 'я', 'и', 'і', 'у', 'о', 'е'
        ]
        endings.sort(key=len, reverse=True)
        
        for e in endings:
            if target.endswith(e):
                if len(target) - len(e) >= 3: 
                    target = target[:-len(e)]
                break
                
        return target

    def get_budget_root(name, lvl):
        words = str(name).replace('_', '-').split() 
        if not words: return ""
        
        target = words[0]
        if lvl == 'l1' and len(words) >= 2:
            target = words[-2]
        elif lvl == 'l2' and len(words) >= 2:
            target = words[-2]
        elif lvl == 'l3' and len(words) >= 2:
            if words[0].lower() == 'бюджет':
                target = words[1]
            else:
                target = words[0]
                
        return normalize_and_chop(target)

    def get_geo_root(name):
        words = str(name).split()
        if not words: return ""
        return normalize_and_chop(words[0])

    def prepare_level(gdf_base, name_col, lvl_id, valid_codes, lvl_name_ua):
        if name_col == 'adm1_name':
            level_gdf = gdf_base.dissolve(by=name_col).reset_index()
        else:
            level_gdf = gdf_base.dissolve(by=['adm1_name', name_col]).reset_index()
            
        level_gdf['display_name'] = level_gdf[name_col]
        
        geo_roots = {row[name_col]: get_geo_root(row[name_col]) for _, row in level_gdf.iterrows()}
        budget_roots = {code: get_budget_root(BUDGET_NAMES[code], lvl_id) for code in valid_codes}

        def get_best_match(geo_row):
            geo_name = geo_row[name_col]
            geo_oblast = geo_row['adm1_name'] 
            geo_clean = geo_roots[geo_name]
            
            if len(geo_clean) < 2: return None
            
            best_code, best_score = None, 0.0
            
            for code in valid_codes:
                # ВИКОРИСТОВУЄМО ALL_OBLASTS
                if ALL_OBLASTS.get(code[:2]) != geo_oblast: continue 
                    
                budget_clean = budget_roots[code]
                if len(budget_clean) < 2: continue
                
                if geo_clean == budget_clean:
                    return code
                elif geo_clean.startswith(budget_clean) and len(budget_clean) >= 4:
                    score = 0.95
                    if score > best_score:
                        best_score, best_code = score, code
                elif budget_clean.startswith(geo_clean) and len(geo_clean) >= 4:
                    score = 0.95
                    if score > best_score:
                        best_score, best_code = score, code
                else:
                    ratio = difflib.SequenceMatcher(None, geo_clean, budget_clean).ratio()
                    if ratio > best_score:
                        best_score, best_code = ratio, code
                    
            return best_code if best_score > 0.65 else None

        level_gdf['BUDGET_CODE'] = level_gdf.apply(get_best_match, axis=1)

        level_gdf = level_gdf[level_gdf['BUDGET_CODE'].notna()].copy()
        if level_gdf.empty: return level_gdf
            
        level_gdf['lon'] = level_gdf.geometry.centroid.x
        level_gdf['lat'] = level_gdf.geometry.centroid.y

        for y in YEARS:
            level_gdf[f'BAL_{y}'] = level_gdf['BUDGET_CODE'].apply(lambda c: AGG_STATS[y].get(c, {}).get('BAL', 0))
            level_gdf[f'TOT_INC_{y}'] = level_gdf['BUDGET_CODE'].apply(lambda c: AGG_STATS[y].get(c, {}).get('TOT_INC', 0))
            level_gdf[f'TOT_EXP_{y}'] = level_gdf['BUDGET_CODE'].apply(lambda c: AGG_STATS[y].get(c, {}).get('TOT_EXP', 0))
            
            for cat in ['INC', 'PROG', 'ECON', 'FUNC']:
                for d in [1, 2, 4, 8]:
                    available_codes = FINAL_UI[y][lvl_id][cat][d].keys() if d in FINAL_UI[y][lvl_id][cat] else []
                    for code in available_codes:
                        level_gdf[f"{cat}_{code}_{y}"] = level_gdf['BUDGET_CODE'].apply(lambda c: AGG_STATS[y].get(c, {}).get(cat, {}).get(code, 0))

        prefixes = tuple(['BAL_', 'TOT_INC_', 'TOT_EXP_', 'INC_', 'PROG_', 'ECON_', 'FUNC_'])
        cols_to_keep = ['geometry', 'display_name', 'BUDGET_CODE', 'lon', 'lat'] + [c for c in level_gdf.columns if c.startswith(prefixes)]
        level_gdf = level_gdf[cols_to_keep]

        for col in level_gdf.columns:
            if col not in ['geometry', 'display_name', 'BUDGET_CODE']:
                level_gdf[col] = pd.to_numeric(level_gdf[col], errors='coerce').fillna(0)
                
        return level_gdf

    hromada_codes = [c for c, lvl in CODE_LEVELS.items() if lvl == 'l3']
    raion_codes = [c for c, lvl in CODE_LEVELS.items() if lvl == 'l2']
    oblast_codes = [c for c, lvl in CODE_LEVELS.items() if lvl == 'l1']

    gdf_level_3 = prepare_level(gdf, 'adm3_name1' if 'adm3_name1' in gdf.columns else 'adm3_name', 'l3', hromada_codes, "Громади (l3)")
    gdf_level_2 = prepare_level(gdf, 'adm2_name', 'l2', raion_codes, "Райони (l2)")
    gdf_level_1 = prepare_level(gdf, 'adm1_name', 'l1', oblast_codes, "Області (l1)")

    # --- ЗБЕРЕЖЕННЯ В КЕШ ---
    print(f"\n💾 Зберігаю підготовлені дані у папку '{CACHE_DIR}'...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    l1_json = gdf_level_1.to_json()
    l2_json = gdf_level_2.to_json()
    l3_json = gdf_level_3.to_json()
    ui_json = json.dumps(FINAL_UI)
    
    with open(os.path.join(CACHE_DIR, 'l1.geojson'), 'w', encoding='utf-8') as f: f.write(l1_json)
    with open(os.path.join(CACHE_DIR, 'l2.geojson'), 'w', encoding='utf-8') as f: f.write(l2_json)
    with open(os.path.join(CACHE_DIR, 'l3.geojson'), 'w', encoding='utf-8') as f: f.write(l3_json)
    with open(os.path.join(CACHE_DIR, 'ui_config.json'), 'w', encoding='utf-8') as f: f.write(ui_json)
