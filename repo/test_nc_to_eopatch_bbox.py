import os
from datetime import datetime
import xarray as xr
import numpy as np
import geopandas as gpd
from eolearn.core import EOPatch, FeatureType
import logging
from sentinelhub.geometry import BBox
import fs.move  

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def create_eopatch_from_nc(nc_file_path, vector_data_path, output_eopatch_dir):
    try:
        LOGGER.info(f'Loading NetCDF file from {nc_file_path}')
        ds = xr.open_dataset(nc_file_path)

        # Create a new EOPatch
        eopatch = EOPatch()

        LOGGER.info('Adding raster data from NetCDF to EOPatch')
        for var in ds.data_vars:
            if var != 'spatial_ref':
                data = ds[var].values
                # Check data dimensions and reshape if necessary
                if data.ndim == 2:
                    data = data[np.newaxis, ..., np.newaxis]  # Add time and channel dimensions
                elif data.ndim == 3:
                    data = data[np.newaxis, ...]  # Add time dimension
                eopatch.add_feature(FeatureType.DATA, var, data)

        LOGGER.info(f'Loading vector data from {vector_data_path}')
        vector_gdf = gpd.read_file(vector_data_path)

        # Add a TIMESTAMP column with the current timestamp
        vector_gdf['TIMESTAMP'] = datetime.now()

        # Add original vector data to EOPatch
        eopatch.add_feature(FeatureType.VECTOR, 'vector_data', vector_gdf)

        # Compute the bounding box from vector data
        bbox_coords = vector_gdf.total_bounds

        # Ensure bbox_coords is in the format (x_min, y_min, x_max, y_max)
        bbox = BBox(bbox=(bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]), crs=vector_gdf.crs)

        # Assign bbox to top-level attribute
        eopatch.bbox = bbox

        # Example to create and add raster masks based on vector data
        resolution = 10  # Resolution in meters
        minx, miny, maxx, maxy = bbox_coords
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        LOGGER.info(f'Creating raster mask of size {width}x{height}')

        # Create raster mask from vector data
        raster_mask = np.zeros((height, width), dtype=np.uint8)
        for geom in vector_gdf.geometry:
            minr, minc = int((geom.bounds[1] - miny) / resolution), int((geom.bounds[0] - minx) / resolution)
            maxr, maxc = int((geom.bounds[3] - miny) / resolution), int((geom.bounds[2] - minx) / resolution)
            raster_mask[minr:maxr, minc:maxc] = 1

        # Add time and channel dimensions to raster mask
        raster_mask = raster_mask[np.newaxis, ..., np.newaxis]
        eopatch.add_feature(FeatureType.MASK, 'raster_mask', raster_mask)

        # Create a unique name for the EOPatch based on current timestamp
        eopatch_name = f'eopatch_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        eopatch_path = os.path.join(output_eopatch_dir, eopatch_name)

        LOGGER.info(f'Saving EOPatch to {eopatch_path}')
        if not os.path.exists(eopatch_path):
            os.makedirs(eopatch_path)
        eopatch.save(eopatch_path)

        return eopatch
    except Exception as e:
        LOGGER.error(f'An error occurred: {e}')
        raise

def main():
    # Paths to files
    nc_file_path = '/home/joseph/Code/localstorage/dataset/sentinel2/images/FR/FR_9334_S2_10m_256.nc'
    vector_data_path = '/home/joseph/Code/localstorage/aoi_geometry.geojson'
    output_eopatch_dir = '/home/joseph/Code/localstorage/eopatchs/'

    # Create EOPatch from NetCDF data
    create_eopatch_from_nc(nc_file_path, vector_data_path, output_eopatch_dir)

if __name__ == "__main__":
    main()
