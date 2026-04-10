import os
import json
import time
import requests
import hashlib

GEMINI_API_KEY = "insert yours"
MODEL_NAME = "gemini-2.0-flash"
BASE_DATA_DIR = "viz_dashboard_data2"

TARGET_DIRS = [
    BASE_DATA_DIR,
    os.path.join(BASE_DATA_DIR, "pca_t_data"),
    os.path.join(BASE_DATA_DIR, "chunk_l3_geo")
]
CONFIG_PATH = os.path.join(BASE_DATA_DIR, "ui_config.json")
OUTPUT_DIR = "eng_names"

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:10]

def get_category_filename(name, code, source=""):
    n_low = str(name).lower()
    if n_low.startswith("група "):
        return "groups.json"
    if source == "PROG":
        return "programmatic.json"
    if source == "FUNC":
        return "functional.json"
    if source == "INC":
        return "incomes.json"
    if source == "ECON":
        return "economic.json"
    code_str = str(code).strip()
    c_len = len(code_str)
    geo_keywords = ["громади", "області", "району", "міської", "сільської", "селищної", "бюджет"]
    if c_len >= 10 or (any(k in n_low for k in geo_keywords) and (c_len == 0 or c_len >= 10)):
        return "geography.json"
    if not code_str:
        return "other.json"
    if c_len == 8:
        return "incomes.json"
    if c_len == 7:
        return "programmatic.json"
    if c_len == 4:
        return "economic.json" if code_str.startswith(("2", "3")) else "functional.json"
    if c_len in [1, 2]:
        return "groups.json"
    return "other.json"

def get_all_terms():
    extracted = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for top_key, top_val in data.items():
                    def walk_config(obj, src):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if str(k).isdigit() and isinstance(v, str):
                                    if any('\u0400' <= c <= '\u04FF' for c in v):
                                        extracted[v.strip()] = {"code": str(k).strip(), "source": src}
                                walk_config(v, src)
                        elif isinstance(obj, list):
                            for i in obj:
                                walk_config(i, src)
                    walk_config(top_val, top_key)
            except Exception:
                pass
    for d in TARGET_DIRS:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for file in files:
                if not file.endswith((".json", ".geojson")):
                    continue
                src = "OTHER"
                if "_PROG" in file:
                    src = "PROG"
                elif "_FUNC" in file:
                    src = "FUNC"
                elif "_ECON" in file:
                    src = "ECON"
                elif "_INC" in file:
                    src = "INC"
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "features" in data:
                            for feature in data["features"]:
                                props = feature.get("properties", {})
                                name = props.get("display_name") or props.get("adm3_name") or props.get("adm1_name")
                                code = props.get("BUDGET_CODE", "")
                                if name and isinstance(name, str):
                                    if any('\u0400' <= c <= '\u04FF' for c in name):
                                        extracted[name.strip()] = {"code": str(code).strip(), "source": src}
                        def walk_data(obj):
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if isinstance(v, str) and " - " in v:
                                        parts = v.split(" - ", 1)
                                        if parts[0].strip().isdigit():
                                            c_val, n_val = parts[0].strip(), parts[1].strip()
                                            if any('\u0400' <= c <= '\u04FF' for c in n_val):
                                                extracted[n_val] = {"code": c_val, "source": src}
                                    elif k == "name" and isinstance(v, str):
                                        c_val = obj.get("code", "")
                                        if any('\u0400' <= c <= '\u04FF' for c in v):
                                            extracted[v.strip()] = {"code": str(c_val).strip(), "source": src}
                                    walk_data(v)
                            elif isinstance(obj, list):
                                for i in obj:
                                    walk_data(i)
                        walk_data(data)
                except Exception:
                    pass
    return extracted

def save_all_dicts(final_dict):
    categorized = {}
    for ukr, data in final_dict.items():
        cat = get_category_filename(ukr, data.get("code", ""), data.get("source", ""))
        clean_data = {
            "ukr": data.get("ukr", ""),
            "eng": data.get("eng", ""),
            "code": data.get("code", "")
        }
        if cat not in categorized:
            categorized[cat] = {}
        categorized[cat][ukr] = clean_data
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".json") and filename != "ui_elements.json":
            os.remove(os.path.join(OUTPUT_DIR, filename))
    for cat, data in categorized.items():
        path = os.path.join(OUTPUT_DIR, cat)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def translate_batch(names_batch):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    hash_to_ukr = {get_hash(name): name for name in names_batch}
    input_dict = {h: ukr for h, ukr in hash_to_ukr.items()}
    prompt = "Translate Ukrainian budget/territory terms to English. Return ONLY a valid JSON object where keys are IDs and values are strings.\n"
    payload = {
        "contents": [{"parts": [{"text": prompt + json.dumps(input_dict, ensure_ascii=False)}] }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print("API Error:", response.status_code, response.text)
            return {}
        txt = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        if txt.startswith("```json"):
            txt = txt.replace("```json", "").replace("```", "").strip()
        elif txt.startswith("```"):
            txt = txt.replace("```", "").strip()
        parsed_data = json.loads(txt)
        raw_results = {}
        if isinstance(parsed_data, list):
            for item in parsed_data:
                if isinstance(item, dict):
                    raw_results.update(item)
        elif isinstance(parsed_data, dict):
            raw_results = parsed_data
        final_results = {}
        for h, eng_text in raw_results.items():
            if h in hash_to_ukr and isinstance(eng_text, str):
                original_ukr = hash_to_ukr[h]
                final_results[original_ukr] = eng_text.strip()
        return final_results
    except Exception as e:
        print("Exception in translate_batch:", e)
        return {}

def main():
    print("Start script...")
    ukr_data = get_all_terms()
    ukr_names = list(ukr_data.keys())
    print("Total unique names:", len(ukr_names))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_dict = {}
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith(".json") and file != "ui_elements.json":
            src = "OTHER"
            if "programmatic" in file:
                src = "PROG"
            elif "functional" in file:
                src = "FUNC"
            elif "economic" in file:
                src = "ECON"
            elif "incomes" in file:
                src = "INC"
            with open(os.path.join(OUTPUT_DIR, file), 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for k, v in data.items():
                        if isinstance(v, dict) and v.get("eng"):
                            if not any('\u0400' <= char <= '\u04FF' for char in v["eng"]):
                                v["source"] = src
                                final_dict[k] = v
                except:
                    pass
    api_queue = []
    local_translated = 0
    for name in ukr_names:
        if name in final_dict:
            final_dict[name]["source"] = ukr_data[name]["source"]
            continue
        if name.lower().startswith("група "):
            parts = name.split(" ", 1)
            if len(parts) == 2:
                final_dict[name] = {
                    "ukr": name,
                    "eng": f"Group {parts[1]}",
                    "code": ukr_data[name]["code"],
                    "source": ukr_data[name]["source"]
                }
                local_translated += 1
        else:
            api_queue.append(name)
    print("Locally translated:", local_translated)
    print("New names sent to API:", len(api_queue))
    if local_translated > 0:
        save_all_dicts(final_dict)
    if not api_queue:
        print("DONE. All data already translated.")
        return
    batch_size = 40
    for i in range(0, len(api_queue), batch_size):
        batch = api_queue[i:i + batch_size]
        print("Processing Batch", i//batch_size + 1, "/", (len(api_queue)-1)//batch_size + 1)
        translations = translate_batch(batch)
        if not translations:
            print("Batch failed. Skipping...")
            time.sleep(5)
            continue
        print("Success, translated:", len(translations))
        for ukr, eng in translations.items():
            if ukr in ukr_data:
                final_dict[ukr] = {
                    "ukr": ukr,
                    "eng": eng,
                    "code": ukr_data[ukr]["code"],
                    "source": ukr_data[ukr]["source"]
                }
        save_all_dicts(final_dict)
        time.sleep(3)
    print("DONE. Files saved in eng_names/")

if __name__ == "__main__":
    main()