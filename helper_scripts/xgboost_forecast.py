import pandas as pd
import numpy as np
import json
import os
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

COMPONENTS = "3_comp" 
DATA_DIR = os.path.join('pca_t_data', COMPONENTS)
OUTPUT_METRICS_DIR = "metrics_output" 

CATEGORIES = ['income', 'prog']
FORECAST_STEP = 1 

os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

def process_category(category):
    clust_file = os.path.join(DATA_DIR, 'clusters', f"clusters_{category}.json")
    traj_file = os.path.join(DATA_DIR, 'trajectories', f"traj_{category}.json")
    
    if not os.path.exists(clust_file) or not os.path.exists(traj_file):
        print(f"Files for {category} not found!")
        return

    with open(clust_file, 'r', encoding='utf-8') as f:
        clust_data = json.load(f)
        
    c_rows = []
    for year, codes in clust_data.get('l3', {}).items():
        for code, info in codes.items():
            c_rows.append({'year': int(year), 'code': code, 'cluster': int(info['cluster'])})
    df_clust = pd.DataFrame(c_rows)

    with open(traj_file, 'r', encoding='utf-8') as f:
        traj_data = json.load(f)
        
    t_rows = []
    for code, info in traj_data.get('trajectories', {}).items():
        if info.get('level') != 'l3': continue
        pc_keys = [k for k in info.keys() if k.startswith('pc')]
        
        for i, date_str in enumerate(info['dates']):
            year = int(date_str.split('.')[1])
            row = {'year': year, 'code': code}
            for pc in pc_keys:
                row[pc] = info[pc][i]
            t_rows.append(row)
            
    df_traj = pd.DataFrame(t_rows)
    df_traj = df_traj.groupby(['code', 'year']).mean().reset_index()

    df = pd.merge(df_clust, df_traj, on=['code', 'year'], how='inner')
    df = df.sort_values(by=['code', 'year'])
    
    pc_cols = [c for c in df.columns if c.startswith('pc')]
    
    df['prev_cluster'] = df.groupby('code')['cluster'].shift(1)
    for pc in pc_cols:
        df[f'prev_{pc}'] = df.groupby('code')[pc].shift(1)

    df['target_cluster'] = df.groupby('code')['cluster'].shift(-FORECAST_STEP)
    
    req_cols = ['target_cluster', 'prev_cluster'] + [f'prev_{pc}' for pc in pc_cols]
    ml_df = df.dropna(subset=req_cols).copy()
    
    ml_df['target_cluster'] = ml_df['target_cluster'].astype(int)
    ml_df['prev_cluster'] = ml_df['prev_cluster'].astype(int)

    feature_cols = ['cluster', 'prev_cluster'] + pc_cols + [f'prev_{pc}' for pc in pc_cols]

    yearly_accuracies = []
    cluster_metrics_train_yearly = []
    cluster_metrics_test_yearly = []

    available_years = sorted(ml_df['year'].unique())
    if len(available_years) < 2: return
    test_years = available_years[1:]
    
    all_y_true, all_y_pred = [], []

    for base_year in test_years:
        target_year = base_year + FORECAST_STEP
        
        train_df = ml_df[ml_df['year'] < base_year]
        test_df = ml_df[ml_df['year'] == base_year]
        
        if train_df.empty or test_df.empty: continue

        le = LabelEncoder()
        train_df['target_encoded'] = le.fit_transform(train_df['target_cluster'])
        valid_test_df = test_df[test_df['target_cluster'].isin(le.classes_)].copy()
        if valid_test_df.empty: continue
            
        valid_test_df['target_encoded'] = le.transform(valid_test_df['target_cluster'])
        
        X_train, y_train = train_df[feature_cols], train_df['target_encoded']
        X_test, y_test = valid_test_df[feature_cols], valid_test_df['target_encoded']
        
        if len(y_train.unique()) <= 1: continue

        model = xgb.XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False,
            random_state=42, n_estimators=100, max_depth=4, learning_rate=0.1
        )
        model.fit(X_train, y_train)
        
        y_train_pred_encoded = model.predict(X_train)
        y_test_pred_encoded = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred_encoded)
        test_acc = accuracy_score(y_test, y_test_pred_encoded)
        
        print(f"Training up to {base_year-1} | FORECAST FOR {target_year} | Test Acc: {test_acc:.2%}")
        
        yearly_accuracies.append({
            'Training_Window': f"Up to {base_year-1} -> {base_year}",
            'Input_Year': base_year,
            'Forecast_Year': target_year,
            'Train_Accuracy': round(train_acc, 4),
            'Test_Accuracy': round(test_acc, 4)
        })

        y_pred_original = le.inverse_transform(y_test_pred_encoded)
        y_test_original = le.inverse_transform(y_test)
        all_y_true.extend(y_test_original)
        all_y_pred.extend(y_pred_original)

        y_test_arr = np.array(y_test_original)
        y_pred_arr = np.array(y_pred_original)
        report_dict_test = classification_report(y_test_original, y_pred_original, output_dict=True, zero_division=0)
        
        for class_label, metrics in report_dict_test.items():
            if class_label.isdigit():
                c = int(class_label)
                correct_for_c = (y_test_arr == c) == (y_pred_arr == c)
                cluster_acc = correct_for_c.mean()

                cluster_metrics_test_yearly.append({
                    'Forecast_Year': target_year,
                    'Cluster': f"Кл. {class_label}",
                    'Accuracy': round(cluster_acc, 2),
                    'Precision': round(metrics['precision'], 2),
                    'Recall': round(metrics['recall'], 2),
                    'F1-Score': round(metrics['f1-score'], 2)
                })

        y_train_pred_original = le.inverse_transform(y_train_pred_encoded)
        y_train_original = le.inverse_transform(y_train)
        
        y_train_arr = np.array(y_train_original)
        y_train_pred_arr = np.array(y_train_pred_original)
        report_dict_train = classification_report(y_train_original, y_train_pred_original, output_dict=True, zero_division=0)
        
        for class_label, metrics in report_dict_train.items():
            if class_label.isdigit():
                c = int(class_label)
                correct_for_c = (y_train_arr == c) == (y_train_pred_arr == c)
                cluster_acc_train = correct_for_c.mean()

                cluster_metrics_train_yearly.append({
                    'Forecast_Year': target_year,
                    'Cluster': f"Кл. {class_label}",
                    'Accuracy': round(cluster_acc_train, 2),
                    'Precision': round(metrics['precision'], 2),
                    'Recall': round(metrics['recall'], 2),
                    'F1-Score': round(metrics['f1-score'], 2)
                })

    if yearly_accuracies:
        pd.DataFrame(yearly_accuracies).to_csv(os.path.join(OUTPUT_METRICS_DIR, f'yearly_accuracy_{category}.csv'), index=False)

    def save_cluster_metrics(metrics_list, dataset_name):
        if not metrics_list: return
        df_clust = pd.DataFrame(metrics_list)
        metrics_to_pivot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for m in metrics_to_pivot:
            pivot_df = df_clust.pivot(index='Forecast_Year', columns='Cluster', values=m)
            if m == 'Accuracy':
                pivot_df = pivot_df.map(lambda x: f"{x*100:.0f}%" if pd.notnull(x) else "")
            
            file_name = f'cluster_{m.lower()}_{dataset_name}_yearly_{category}.csv'
            pivot_df.to_csv(os.path.join(OUTPUT_METRICS_DIR, file_name))

    save_cluster_metrics(cluster_metrics_train_yearly, 'train')
    save_cluster_metrics(cluster_metrics_test_yearly, 'test')
    
    print(f"Saved cluster metrics (Train and Test) in: {OUTPUT_METRICS_DIR}")

for cat in CATEGORIES:
    process_category(cat)