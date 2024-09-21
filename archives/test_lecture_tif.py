import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

def display_tif_info(file_path):
    with rasterio.open(file_path) as src:
        # Afficher les méta-données du fichier
        print("Meta data:", src.meta)

        # Afficher les informations sur les bandes
        print("Band count:", src.count)
        print("Width:", src.width)
        print("Height:", src.height)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Bounds:", src.bounds)

        # Afficher les statistiques de chaque bande
        for i in range(1, src.count + 1):
            band = src.read(i)
            print(f"Statistics for band {i}:")
            print(f"  Min: {band.min()}")
            print(f"  Max: {band.max()}")
            print(f"  Mean: {band.mean()}")
            print(f"  Std: {band.std()}")

        # Afficher une visualisation rapide de la première bande
        show(src, 1)
        
        # Afficher les indices des bandes (valeurs de pixels)
        print("\nIndices des bandes (valeurs de pixels):")
        for i in range(1, src.count + 1):
            band = src.read(i)
            print(f"Band {i} values:")
            print(band)
            
# Utiliser la fonction avec le chemin de votre fichier TIFF
# tif_file_path = '/home/joseph/Code/localstorage/dataset/orthophoto/images/FR/FR_9334_ortho_1m_512.tif'
# tif_file_path = '/home/joseph/Code/localstorage/dataset/orthophoto/masks/FR/FR_9334_ortholabel_1m_512.tif'
# tif_file_path = '/home/joseph/Code/localstorage/tiffstorage/aoi.tiff'
tif_file_path = '/home/joseph/Code/localstorage/dataset/sentinel2/masks/FR/FR_9334_S2label_10m_256.tif'
display_tif_info(tif_file_path)
