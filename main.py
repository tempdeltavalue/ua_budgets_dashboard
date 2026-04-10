from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import os
import warnings
import json
from fastapi.staticfiles import StaticFiles
from helper_scripts.utils import load_and_prepare_data
from fastapi.middleware.gzip import GZipMiddleware

warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

base_data_path = "viz_dashboard_data2"
PCA_BASE_DIR = os.path.join(base_data_path, "pca_t_data")

server_state = {
    "geo_data": {"l1": {}, "l2": {}, "l3": {}},
    "ui_config": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_state
    server_state = load_and_prepare_data(server_state, CACHE_DIR=base_data_path)
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/old_dashboard", StaticFiles(directory="extra_static_html", html=True), name="old_dashboard")
app.mount("/static", StaticFiles(directory="web_ui"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    file_path = os.path.join("web_ui", "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Помилка: index.html не знайдено в папці web_ui</h1>", status_code=404)

@app.get("/compare", response_class=HTMLResponse)
async def serve_compare():
    file_path = os.path.join("web_ui", "compare.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Помилка: compare.html не знайдено в папці web_ui</h1>", status_code=404)

@app.get("/pca_compare", response_class=HTMLResponse)
async def serve_pca_compare():
    file_path = os.path.join("web_ui", "pca_compare.html")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Помилка: pca_compare.html не знайдено в папці web_ui</h1>", status_code=404)

@app.get("/{filename}")
async def serve_root_files(filename: str):
    file_path = os.path.join("web_ui", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(content={"error": f"File {filename} not found"}, status_code=404)
    
# --- ОПТИМІЗОВАНІ РОУТИ ДЛЯ НОВОЇ СТРУКТУРИ ---

@app.get("/api/pca_data/{category}")
async def get_pca_data(category: str, n_comp: int = 3):
    file_path = os.path.join(PCA_BASE_DIR, f"{n_comp}_comp", "trajectories", f"traj_{category}.json")
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return FileResponse(
            file_path, 
            media_type="application/json",
            headers={"X-File-Size": str(file_size)}
        )
    else:
        return JSONResponse(content={"error": f"PCA data file not found for {category} ({n_comp} comp)"}, status_code=404)

@app.get("/api/kmeans_data/{category}")
async def get_kmeans_data(category: str, n_comp: int = 3):
    file_path = os.path.join(PCA_BASE_DIR, f"{n_comp}_comp", "clusters", f"clusters_{category}.json")
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return FileResponse(
            file_path, 
            media_type="application/json",
            headers={"X-File-Size": str(file_size)}
        )
    else:
        return JSONResponse(content={"error": f"KMeans data file not found for {category} ({n_comp} comp)"}, status_code=404)

@app.get("/api/pca_errors/{category}")
async def get_pca_errors(category: str, n_comp: int = 3):
    file_path = os.path.join(PCA_BASE_DIR, f"{n_comp}_comp", "errors", f"error_{category}.json")
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return FileResponse(
            file_path, 
            media_type="application/json",
            headers={"X-File-Size": str(file_size)}
        )
    else:
        return JSONResponse(content={"error": f"PCA error file not found for {category} ({n_comp} comp)"}, status_code=404)
    
@app.get("/api/config")
async def get_config():
    file_path = os.path.join(base_data_path, "ui_config.json")
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return FileResponse(
            file_path, 
            media_type="application/json",
            headers={"X-File-Size": str(file_size)}
        )
    return JSONResponse(content=server_state.get("ui_config", {}))

@app.get("/api/filters_config")
async def get_filters_config():
    """ Віддає повну конфігурацію всіх фільтрів для UI """
    return JSONResponse(content={
        "view_modes": [
            {"value": "data", "label": "Raw Data"},
            {"value": "index", "label": "Analytical Indexes"}
        ],
        "levels": [
            {"value": "l3", "label": "Communities (Hromadas)"},
            {"value": "l2", "label": "Districts (Raions)"},
            {"value": "l1", "label": "Regions (Oblasts)"}
        ],
        "categories": [
            {"value": "BAL", "label": "Total Balance"},
            {"value": "INC", "label": "Own Revenues"},
            {"value": "PROG", "label": "Program Expenses (KPKVK)"},
            {"value": "ECON", "label": "Economic Expenses (KEKV)"},
            {"value": "FUNC", "label": "Functional Expenses (FKVK)"}
        ],
        "detail_levels": [
            {"value": "1", "label": "Groups (1 digit)"},
            {"value": "2", "label": "Subgroups (2 digits)"},
            {"value": "4", "label": "Articles (4 digits)", "selected": True},
            {"value": "8", "label": "Maximum (8 digits)"}
        ],
        "components": [
            {"value": "3", "label": "3 Components (Base)"},
            {"value": "10", "label": "10 Components (Deep)"}
        ],
        "analytical_indexes": [
            {"value": "income", "label": "PCA KMeans Income"},
            {"value": "prog", "label": "PCA KMeans Program Expenses"},
            {"value": "econ", "label": "PCA KMeans Economic Expenses"},
            {"value": "func", "label": "PCA KMeans Functional Expenses"}
        ],
        "pca_errors": [
            {"value": "income_error", "label": "PCA Variance (Income)"},
            {"value": "prog_error", "label": "PCA Variance (Program Exp)"},
            {"value": "econ_error", "label": "PCA Variance (Economic Exp)"},
            {"value": "func_error", "label": "PCA Variance (Functional Exp)"}
        ]
    })

@app.get("/api/geo/{level}")
async def get_geo_data(level: str):
    # Спочатку припускаємо стандартну назву
    file_path = os.path.join(base_data_path, f"{level}.geojson")
    
    # Якщо це l3, пробуємо знайти оптимізовану базу
    if level == "l3":
        base_path = os.path.join(base_data_path, "l3_base.geojson")
        # Якщо оптимізований файл існує, беремо його. Якщо ні - залишається l3.geojson
        if os.path.exists(base_path):
            file_path = base_path
            
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return FileResponse(
            file_path, 
            media_type="application/json",
            headers={"X-File-Size": str(file_size)}
        )
    return JSONResponse(content={"error": "Level not found"}, status_code=404)

@app.get("/api/chunk_data/{level}/{category}")
async def get_chunk_data(level: str, category: str):
    file_path = os.path.join(base_data_path, f"chunk_{level}_geo", f"{level}_{category}.json")
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        return FileResponse(
            file_path, 
            media_type="application/json",
            headers={"X-File-Size": str(file_size)}
        )
    return JSONResponse(content={"error": "Chunk not found"}, status_code=404)

@app.get("/api/compare_data")
async def get_compare_data(items: str):
    selected_features = []
    if not items:
        return JSONResponse(content={"features": selected_features})
        
    pairs = items.split(",")
    for pair in pairs:
        if ":" not in pair: continue
        level, code = pair.split(":")
        
        if level in server_state["geo_data"]:
            features = server_state["geo_data"][level].get("features", [])
            for f in features:
                if f["properties"].get("BUDGET_CODE") == code:
                    feature_copy = dict(f)
                    feature_copy["properties"] = dict(f["properties"])
                    feature_copy["properties"]["_level"] = level
                    selected_features.append(feature_copy)
                    break 
                    
    return JSONResponse(content={"features": selected_features})

@app.get("/api/translations/{filename}")
async def get_translation(filename: str):
    allowed_files = [
        "economic.json", "functional.json", "geography.json", 
        "groups.json", "incomes.json", "programmatic.json", "other.json", "ui_elements.json"
    ]
    
    if filename not in allowed_files:
        return JSONResponse(content={"error": "Invalid translation file requested"}, status_code=403)
        
    file_path = os.path.join("web_ui", "eng_names", filename)
    
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/json")
    else:
        return JSONResponse(content={"error": f"Translation file {filename} not found"}, status_code=404)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)