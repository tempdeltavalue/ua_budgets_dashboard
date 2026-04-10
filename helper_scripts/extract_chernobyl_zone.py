import geopandas as gpd
import os

def extract_chernobyl():
    input_path = "ukr_admin_boundaries/ukr_admin3.geojson"
    output_path = "viz_dashboard_data2/chernobyl.geojson"

    print("[1/3] Searching for Chernobyl zone...")
    try:
        gdf = gpd.read_file(input_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    mask = gdf['adm3_name'].str.contains('Chernobyl|Чорнобиль|відчуження', case=False, na=False) | \
           gdf['adm2_name'].str.contains('Chernobyl|Чорнобиль', case=False, na=False)
    
    chern_gdf = gdf[mask].copy()

    if chern_gdf.empty:
        print("Error: Zone not found by name. Retrying with coordinates...")
        chern_gdf = gdf[(gdf.geometry.centroid.x > 29.7) & (gdf.geometry.centroid.x < 30.3) & 
                        (gdf.geometry.centroid.y > 51.2) & (gdf.geometry.centroid.y < 51.4)].copy()

    print("[2/3] Processing zone attributes...")
    chern_gdf['isExclusionZone'] = True
    chern_gdf['isOccupied'] = False
    chern_gdf['name'] = "Chernobyl Exclusion Zone"

    print("[3/3] Simplifying boundaries (0.005)...")
    chern_gdf['geometry'] = chern_gdf['geometry'].simplify(0.005, preserve_topology=True)
    chern_gdf = chern_gdf[['name', 'isOccupied', "isExclusionZone", 'geometry']]

    os.makedirs("viz_dashboard_data2", exist_ok=True)
    chern_gdf.to_file(output_path, driver="GeoJSON")
    print(f"Done. File saved: {output_path}")

if __name__ == "__main__":
    extract_chernobyl()