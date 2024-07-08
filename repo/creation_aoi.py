import rasterio
import geopandas as gpd
from shapely.geometry import box
import json

def tif_to_geometry(tif_files):
    geometries = []
    
    for tif in tif_files:
        with rasterio.open(tif) as src:
            bounds = src.bounds
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            geometries.append(geom)
    
    return geometries

def create_aoi_geojson(geometries, output_geojson_path):
    aoi_gdf = gpd.GeoDataFrame({'geometry': geometries}, crs="EPSG:3035")
    aoi_union = aoi_gdf.unary_union
    aoi_gdf = gpd.GeoDataFrame({'geometry': [aoi_union]}, crs="EPSG:3035")
    aoi_gdf.to_file(output_geojson_path, driver='GeoJSON')

def create_gpkg_with_parcels(parcels, output_gpkg_path):
    parcels_gdf = gpd.GeoDataFrame(parcels, crs="EPSG:3035")
    parcels_gdf.to_file(output_gpkg_path, driver='GPKG')

# Test d'utilisation :
tif_files = [f'/home/joseph/Code/localstorage/dataset/orthophoto/images/FR/FR_{i}_ortho_1m_512.tif' for i in [2149, 2968, 3251, 3803, 4079, 4367, 5171, 5173, 5463, 6828, 7939, 7940, 8198, 8209, 8478, 8766, 9060, 9314, 9334, 9609, 10130, 10139, 10691, 11231, 11246, 11256, 11390, 11511, 11665, 11791, 12056, 12063]]  # Liste des fichiers TIFF
output_geojson_path = '/home/joseph/Code/localstorage/aoi_geometry.geojson'
output_gpkg_path = '/home/joseph/Code/localstorage/parcel_boundaries.gpkg'

# Extraire les géométries des fichiers TIFF
geometries = tif_to_geometry(tif_files)

# Créer et sauvegarder le fichier GeoJSON avec l'AOI
create_aoi_geojson(geometries, output_geojson_path)

# Exemple de données de parcelles pour le fichier GPKG
# Remplacez cette partie avec vos propres données de parcelles
parcels = {
    'geometry': [box(3358213.7485779882, 2935561.0376300616, 3358725.7516223714, 2936073.0406744448)],
    'parcel_id': [1]
}

# Créer et sauvegarder le fichier GPKG avec les parcelles
create_gpkg_with_parcels(parcels, output_gpkg_path)
