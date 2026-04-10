import os
import requests
import pandas as pd
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ALL_OBLASTS = {
    '01': 'crimea',
    '02': 'vinnytska',
    '03': 'volynska',
    '04': 'dnipropetrovska',
    '05': 'donetska',
    '06': 'zhytomyrska',
    '07': 'zakarpatska',
    '08': 'zaporizka',
    '09': 'ivano_frankivska',
    '10': 'kyivska',
    '11': 'kirovohradska',
    '12': 'luhanska',
    '13': 'lvivska',
    '14': 'mykolaivska',
    '15': 'odeska',
    '16': 'poltavska',
    '17': 'rivnenska',
    '18': 'sumska',
    '19': 'ternopilska',
    '20': 'kharkivska',
    '21': 'khersonska',
    '22': 'khmelnytska',
    '23': 'cherkaska',
    '24': 'chernivetska',
    '25': 'chernihivska',
    '26': 'kyiv_city',
    '27': 'sevastopol_city'
}

class OpenBudgetClient:
    def __init__(self):
        self.BASE_URL = "https://api.openbudget.gov.ua"
        self.API_ENDPOINT = "/api/public/localBudgetData"
        self.PERIOD = "MONTH"
        self.FETCH_YEARS = range(2018, 2026)

        self.BUDGET_ITEMS = [
            "EXPENSES", "INCOMES", "FINANCING_DEBTS", "FINANCING_CREDITOR", "CREDITS"
        ]
        self.CLASSIFICATION_TYPES = [
            "EMPTY", "PROGRAM", "FUNCTIONAL", "ECONOMIC", "CREDIT",
        ]

        self.VALID_COMBINATIONS = {
            ("EXPENSES", "PROGRAM"), 
            ("EXPENSES", "FUNCTIONAL"), 
            ("EXPENSES", "ECONOMIC"),
            ("INCOMES", "EMPTY"),
        }

        self.headers = {
            'Authority': 'api.openbudget.gov.ua',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
        }

        self.session = requests.Session()
        retry_strategy = Retry(
            total=0, 
            backoff_factor=1,
            status_forcelist=[400, 429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_url(self):
        return f"{self.BASE_URL}{self.API_ENDPOINT}"

    def get_file_path(self, base_folder, item, year, classification):
        leaf_folder_name = f"{classification.lower()}_{item.lower()}"
        filename = f"{classification.lower()}_{item.lower()}_{year}.csv"
        folder_path = os.path.join(base_folder, leaf_folder_name)
        return os.path.join(folder_path, filename)

    def fetch_data(self, budget_code, year, item, classification):
        params = {
            "budgetCode": budget_code,
            "budgetItem": item,
            "period": self.PERIOD,
            "year": year,
            "format": "csv"
        }
        if classification != "EMPTY":
            params["classificationType"] = classification

        try:
            print(f"   Downloading: {item:<18} [{classification:<10}] {year}...", end=" ", flush=True)

            response = self.session.get(
                self.get_url(),
                params=params,
                headers=self.headers,
                timeout=(30, 90), 
                verify=False
            )
            response.encoding = 'utf-8'

            if response.status_code != 200:
                print(f"Failed (HTTP {response.status_code}).")
                return None

            if not response.text.strip():
                print("Empty.")
                return None

            print("Ok.")
            return response.text

        except Exception as e:
            print(f" Error: {str(e)[:100]}...") 
            return None

    def save_data(self, text_data, full_path):
        folder = os.path.dirname(full_path)
        os.makedirs(folder, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8-sig') as f:
            f.write(text_data)

    def process_budget(self, budget_code, budget_name, target_folder):
        print("-" * 70)
        print(f"PROCESSING: {budget_name} (Code: {budget_code})")
        print(f"Target: {target_folder}")
        print("-" * 70)

        for classification in self.CLASSIFICATION_TYPES:
            for item in self.BUDGET_ITEMS:
                if (item, classification) not in self.VALID_COMBINATIONS:
                    continue

                for year in self.FETCH_YEARS:
                    full_path = self.get_file_path(target_folder, item, year, classification)

                    if os.path.exists(full_path):
                        continue

                    data = self.fetch_data(budget_code, year, item, classification)
                    
                    if data:
                        self.save_data(data, full_path)
                    
                    time.sleep(0.5) 

def load_all_budgets(file_path):
    print(f"Reading directory: {file_path}...")
    try:
        df = pd.read_excel(file_path, header=8)
        df = df.dropna(subset=['Код бюджету 4'])
        
        df['Код бюджету 4'] = df['Код бюджету 4'].astype(str).str.strip()
        
        budgets = {}
        for _, row in df.iterrows():
            code = row['Код бюджету 4']
            
            if len(code) >= 2:
                region_prefix = code[:2]
                
                if region_prefix in ALL_OBLASTS:
                    budget_type = str(row['Ознака бюджету 3']).strip()
                    
                    if budget_type in ['o', 'r', 'gs', 'gss', 'gm']:
                        name = str(row['Найменування бюджету']).strip()
                        safe_name = re.sub(r'[\\/*?:"<>|]', "", name) 
                        
                        if budget_type == 'o':
                            group_folder = "1_Regional_Budgets"
                        elif budget_type == 'r':
                            group_folder = "2_District_Budgets"
                        else:
                            group_folder = "3_Community_Budgets"
                            
                        region_folder_name = ALL_OBLASTS[region_prefix]
                            
                        budgets[safe_name] = {
                            'code': code,
                            'group_folder': group_folder,
                            'region_folder': region_folder_name
                        }
            
        print(f"Found {len(budgets)} unique budgets across Ukraine.")
        return budgets
        
    except Exception as e:
        print(f"Processing error: {e}")
        return {}

if __name__ == "__main__":
    client = OpenBudgetClient()

    EXCEL_FILE_PATH = "Довідник місцевих бюджетів (станом на 01_01_2026).xlsx"
    ROOT_OUTPUT_DIR = "b_data"

    TARGET_BUDGETS = load_all_budgets(EXCEL_FILE_PATH)

    if not TARGET_BUDGETS:
        print("Halt: No budgets found for download. Check filename.")
        exit()

    for name, data in TARGET_BUDGETS.items():
        code = data['code']
        group = data['group_folder']
        region_dir = data['region_folder']
        
        target_path = os.path.join(ROOT_OUTPUT_DIR, region_dir, group, f"{name}_{code}")
        
        try:
            client.process_budget(code, name, target_path)
        except KeyboardInterrupt:
            print("\n Stopped by user.")
            break
        except Exception as e:
            print(f"CRITICAL ERROR processing {name}: {e}")

    print("\nAll data successfully downloaded or updated!")