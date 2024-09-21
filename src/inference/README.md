<br />

<div align="center">
  <h1 align="center">Niva Project</h1>
  <p align="center">
    Inference pipeline for field delineation using the ResUnet-A model.
  </p>
</div>

## Prerequisites

The same as in https://github.com/Jodp1905/Niva_Project.git plus changes
with upgrade of shapely == 2.0.0 and several packages for tile download:
```sh
pip install pystac_client
pip install odc-stac
pip install folium
pip install geojson
pip install odc-algo
pip install odc-ui
```
## Pipeline

First is tile download, then running of the script on the download tile.

### Tile download

tile_download.ipynb notebook searches, visualizes and downloads the chosen tile.

### Inference pipeline

Prerequisites: providing necessary paths to working directory (PROJECT_DATA_ROOT), saved tile path from previous step (TILE_PATH),
model checkpoints folder (MODEL_FOLDER), path to model's configs (MODEL_CONFIG_PATH).
After that run the code with line below. The output GeoJson will be saved to path CONTOURS_DIR/{version}/*.json.
```sh
python3 inference_pipeline.py
```


