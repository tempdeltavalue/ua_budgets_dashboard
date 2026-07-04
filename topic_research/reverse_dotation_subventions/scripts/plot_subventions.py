import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import textwrap
import os
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default')

data_dir = "article_data/data"
plots_dir = "article_data/plots"
os.makedirs(plots_dir, exist_ok=True)

print("Plotting Subventions Dynamics...")
df_sub = pd.read_csv('article_data/subventions.txt')
df_sub['Код'] = df_sub['Код'].astype(str).str.strip()
df_sub['Категорія'] = df_sub['Категорія'].str.strip()

totals_csv = os.path.join(data_dir, 'subventions_totals.csv')
df_totals = pd.read_csv(totals_csv)
years = df_totals['Рік']

categories = df_sub['Категорія'].unique()
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']

for cat in categories:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    cat_subs = df_sub[df_sub['Категорія'] == cat].drop_duplicates(subset=['Код', 'Найменування'])
    
    for i, (_, row) in enumerate(cat_subs.iterrows()):
        code = row['Код']
        name = row['Найменування']
        full_name = f"[{code}] {name}"
        
        if full_name in df_totals.columns:
            values_in_bln = df_totals[full_name] / 1e9
            
            # Check if there is any non-NaN value
            if values_in_bln.notna().sum() > 0:
                wrapped_name = textwrap.fill(full_name, width=60)
                ax.plot(years, values_in_bln, marker=markers[i % len(markers)], linewidth=2, label=wrapped_name)
        
    ax.set_title(f'Динаміка субвенцій: {cat} (тільки діючі, 2022-2025)', fontsize=16)
    ax.set_xlabel('Рік', fontsize=12)
    ax.set_ylabel('Сума (Мільярди грн)', fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xticks(years)
    
    # Legend below
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('white')
    
    plt.tight_layout()
    safe_cat = "".join([c if c.isalnum() else "_" for c in cat])
    plot_path = os.path.join(plots_dir, f'subventions_category_{safe_cat}_dynamics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved plot to {plot_path}")

print("Subventions plots generated.")
