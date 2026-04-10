# ua_budgets_dashboard
Live Interactive Dashboard: https://ua-budgets-dash.duckdns.org/

![Dashboard Preview](https://github.com/user-attachments/assets/65c4d01a-55ad-4cdf-a66e-7a1ffa55fc3c)

---

## Data Sources & Availability
The analytical framework integrates data from three primary pillars:

1. **Raw Financial Records:** Retrieved from the official state portal: https://openbudget.gov.ua/en. This is the primary source for all municipal revenue and expenditure figures.
2. **Budget Directories:** Classification Codes provided by the Ministry of Finance of Ukraine: https://mof.gov.ua/uk/dovidnyky-misc-budg (used for mapping numeric codes to human-readable categories).
3. **Geospatial Data:** Subnational Administrative Boundaries (Admin 3) provided by OCHA (Humanitarian Data Exchange): https://data.humdata.org/dataset/cod-ab-ukr.

### Experimental Dataset
For convenience, a pre-downloaded and processed dataset is available via Google Drive:
* **Download Parquet Data:** https://drive.google.com/file/d/1K_g-N8xbiAkVdVhMqRYUVWksHcnjmD1-/view?usp=drive_link

> **Storage Requirement:** To run the full pipeline, raw data must be converted to `.csv`. The total dataset requires approximately **20 GB** of local storage.

---

## Analytical Pipeline & Execution Order
This repository is organized as a modular suite of scripts. To reproduce the analysis and dashboard files, execute them in the following sequence:

### 1. Data Processing & Dimensionality Reduction
1. [`download_data.py`](helper_scripts/download_data.py)  
   *Automates API requests to pull multi-year financial records from OpenBudget.*
2. [`process_pca.py`](helper_scripts/process_pca.py)  
   *Applies Principal Component Analysis (PCA) to extract latent fiscal features from high-dimensional budget data.*
3. [`calc_pca_clusters.py`](helper_scripts/calc_pca_clusters.py)  
   *Segments territorial communities into macroeconomic profiles using the k-means algorithm.*
4. [`calculate_clusters_rating.py`](helper_scripts/calculate_clusters_rating.py)  
   *Computes municipal financial health ratings and autonomy scores based on cluster centroids.*

### 2. Forecasting & Research
5. [`xgboost_forecast.py`](helper_scripts/xgboost_forecast.py)  
   *Trains Gradient Boosting models to predict future budget trajectories and structural shifts.*
6. [`clusters_dynamics.ipynb`](helper_scripts/clusters_dynamics.ipynb)  
   *Interactive notebook for analyzing how communities migrate between fiscal clusters over time.*

### 3. Geospatial Synthesis & Dashboard Generation
7. [`extract_chernobyl_zone.py`](helper_scripts/extract_chernobyl_zone.py)  
   *Specialized script for processing the administrative boundaries of the Exclusion Zone.*
8. [`generate_dash_files.py`](helper_scripts/generate_dash_files.py)  
   *Final aggregation step: compiles all outputs into optimized JSON/GeoJSON for the web dashboard.*

---

## Technical Stack & Utilities
* **Core:** Python (Pandas, NumPy, Scikit-learn, XGBoost)
* **Geospatial:** GeoPandas, Shapely
* **Utility:** [`utils.py`](helper_scripts/utils.py) - contains shared helper functions, data cleaning routines, and logging configurations used across the entire pipeline.
