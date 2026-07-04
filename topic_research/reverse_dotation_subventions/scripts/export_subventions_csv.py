import json
import pandas as pd

with open('article_data/data/all_subventions_parsed.json', 'r', encoding='utf-8') as f:
    subs = json.load(f)

# Deduplicate by code and name
unique_subs = []
seen = set()
for s in subs:
    key = (s['code'], s['name'])
    if key not in seen:
        unique_subs.append(s)
        seen.add(key)

categories = {
    'Освіта': ['освіт', 'педагог', 'школ', 'інклюзив', 'навчан', 'учител'],
    'Медицина': ['медич', 'охорон здоров', 'лікар', 'швидк', 'інсулін', 'допомог', 'здоров', 'фап', 'амбулатор', 'лікарн'],
    'Соціальний захист': ['соціал', 'діт', 'сиріт', 'житл', 'компенсаці', 'пільг', 'ветера', 'інвалід', 'сім', 'чорнобил'],
    'Інфраструктура та розвиток': ['інфраструктур', 'будівництв', 'дорог', 'метро', 'транспорт', 'проект', 'розвит', 'ремонт', 'комунал', 'фонд регіон', 'водо', 'смітт'],
    'Війна та безпека': ['війн', 'відновлен', 'наслідків', 'агресії', 'укритт', 'безпечн', 'оповіщен', 'збройн', 'оборон'],
    'Культура та спорт': ['культур', 'спорт', 'мистец', 'басейн', 'палац'],
}

def assign_category(name):
    name_lower = name.lower()
    if 'виключено' in name_lower:
        return 'Історичні (Виключені)'
        
    for cat, keywords in categories.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return 'Інші (Загальні/Некласифіковані)'

# Add category to each subvention
for s in unique_subs:
    s['Група'] = assign_category(s['name'])

# Convert to DataFrame
df = pd.DataFrame(unique_subs)

# Rename columns for convenience and sort
df = df.rename(columns={'code': 'Код', 'name': 'Найменування'})
df = df[['Код', 'Найменування', 'Група']]
df = df.sort_values(by=['Група', 'Код'])

# Save to CSV
csv_path = 'article_data/data/all_subventions_list.csv'
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print(f"Saved {len(df)} subventions to {csv_path}")
