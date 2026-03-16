import geopandas as gpd
import os

def extract_chernobyl():
    input_path = "ukr_admin_boundaries/ukr_admin3.geojson"
    output_path = "viz_dashboard_data2/chernobyl.geojson"

    print("🗺️ [1/3] Шукаю Чорнобильську зону на карті...")
    try:
        gdf = gpd.read_file(input_path)
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return

    # Шукаємо за ключовими словами в назвах громад або районів
    # В адмін-поділі це зазвичай 'Chernobyl' або 'Чорнобиль'
    mask = gdf['adm3_name'].str.contains('Chernobyl|Чорнобиль|відчуження', case=False, na=False) | \
           gdf['adm2_name'].str.contains('Chernobyl|Чорнобиль', case=False, na=False)
    
    chern_gdf = gdf[mask].copy()

    if chern_gdf.empty:
        print("❌ Помилка: Не вдалося знайти зону відчуження за назвою. Спробую за координатами...")
        # Якщо за назвою не знайшло (буває в різних версіях карт), візьмемо прямокутником
        # Приблизні координати зони: 29.5-30.5 довгота, 51.1-51.5 широта
        chern_gdf = gdf[(gdf.geometry.centroid.x > 29.7) & (gdf.geometry.centroid.x < 30.3) & 
                        (gdf.geometry.centroid.y > 51.2) & (gdf.geometry.centroid.y < 51.4)].copy()

    print("✂️ [2/3] Об'єдную полігони зони...")
    chern_gdf['isExclusionZone'] = True  # Новий прапор
    chern_gdf['isOccupied'] = False      # Чітко вказуємо, що не окуповано
    chern_gdf['name'] = "Чорнобильська зона відчуження"
    # chern_gdf = chern_gdf.dissolve(by='isOccupied').reset_index()

    print("📐 [3/3] Згладжую кордони (0.005)...")
    chern_gdf['geometry'] = chern_gdf['geometry'].simplify(0.005, preserve_topology=True)
    chern_gdf = chern_gdf[['name', 'isOccupied', "isExclusionZone", 'geometry']]

    os.makedirs("viz_dashboard_data2", exist_ok=True)
    chern_gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Готово! Файл збережено: {output_path}")

if __name__ == "__main__":
    extract_chernobyl()