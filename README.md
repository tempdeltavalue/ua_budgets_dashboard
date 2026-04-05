# ua_budgets_dashboard
[Live Interactive Dashboard](https://ua-budgets-dash.duckdns.org/)

![Dashboard Preview](https://github.com/user-attachments/assets/65c4d01a-55ad-4cdf-a66e-7a1ffa55fc3c)

## Data Availability
The raw municipal budget data can be retrieved from the official Ukrainian state portal [OpenBudget](https://openbudget.gov.ua/en).

For experimental convenience, the pre-downloaded dataset in `.parquet` format is available here:
* **[Download Parquet Data (Google Drive)](https://drive.google.com/file/d/1K_g-N8xbiAkVdVhMqRYUVWksHcnjmD1-/view?usp=drive_link)**

> **Important:** To run the analytical pipeline, the downloaded `.parquet` files must be converted to `.csv` format. Once extracted and converted, the dataset will require approximately **20 GB** of local storage space. While this dataset is provided for reproducibility, we highly encourage researchers to pull the latest data directly from the official OpenBudget portal for new experiments.

## Analytical Pipeline & Execution Order
This repository is structured as a modular suite of scripts. To reproduce the analysis, clustering, and scoring, execute the scripts in the following sequence:

1. [`download_data.py`](helper_scripts/download_data.py)  
   *Downloads and aggregates municipal financial records from the source.*
2. [`process_pca.py`](helper_scripts/process_pca.py)  
   *Performs dimensionality reduction (PCA) to extract latent fiscal features.*
3. [`calc_pca_clusters.py`](helper_scripts/calc_pca_clusters.py)  
   *Groups territorial communities into macroeconomic profiles using the k-means algorithm.*
4. [`calculate_clusters_rating.py`](helper_scripts/calculate_clusters_rating.py)  
   *Computes the municipal scoring and financial independence ratings based on centroid coordinates.*
