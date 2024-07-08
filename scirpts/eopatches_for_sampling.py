import os
from datetime import datetime
import xarray as xr
import numpy as np
import geopandas as gpd
from eolearn.core import EOPatch, FeatureType
import logging
from sentinelhub.geometry import BBox
from dateutil.tz import tzlocal
import rasterio
import fs.move

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def create_eopatch_from_nc_and_tiff(nc_file_path, vector_data_path, mask_tiff_path, output_eopatch_dir):
    try:
        LOGGER.info(f'Loading NetCDF file from {nc_file_path}')
        ds = xr.open_dataset(nc_file_path)

        LOGGER.info(f'Dataset structure: {ds}')

        # Extracting time data
        time_data = ds['time'].values

        # Convert numpy.datetime64 to ISO 8601 strings
        time_strings = np.datetime_as_string(time_data, unit='ms')  # Adjust unit if necessary

        # Transform time_strings into datetime objects with local timezone
        transformed_timestamps = []
        for t in time_strings:
            dt = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f')
            dt_with_tz = dt.replace(tzinfo=tzlocal())
            transformed_timestamps.append(dt_with_tz)

        # Create a new EOPatch
        eopatch = EOPatch()

        # Initialize a dictionary to group bands and other data
        data_dict = {
            'BANDS': [],
            'CLP': None
        }

        LOGGER.info('Adding raster data from NetCDF to EOPatch')
        for var in ds.data_vars:
            if var != 'spatial_ref':
                data = ds[var].values

                # Check data dimensions and reshape if necessary
                if data.ndim == 2:
                    data = data[np.newaxis, ..., np.newaxis]  # Add time and channel dimensions
                elif data.ndim == 3:
                    data = data[..., np.newaxis]  # Add channel dimension
                elif data.ndim != 4:
                    LOGGER.warning(f"Variable {var} has unexpected number of dimensions: {data.ndim}. Skipping.")
                    continue

                # Group bands into a single BANDS feature
                if var.startswith('B'):
                    data_dict['BANDS'].append(data)
                elif var == 'CLP':
                    data_dict['CLP'] = data
                else:
                    eopatch.add_feature(FeatureType.DATA, var, data)

        # Stack bands data along the last dimension (channels)
        if data_dict['BANDS']:
            bands_data = np.concatenate(data_dict['BANDS'], axis=-1)
            eopatch.add_feature(FeatureType.DATA, 'BANDS', bands_data)

        if data_dict['CLP'] is not None:
            eopatch.add_feature(FeatureType.DATA, 'CLP', data_dict['CLP'])

        LOGGER.info(f'Loading vector data from {vector_data_path}')
        vector_gdf = gpd.read_file(vector_data_path)

        # Compute the bounding box from vector data
        bbox_coords = vector_gdf.total_bounds

        # Ensure bbox_coords is in the format (x_min, y_min, x_max, y_max)
        bbox = BBox(bbox=(bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]), crs=vector_gdf.crs)

        # Assign bbox to top-level attribute
        eopatch.bbox = bbox

        # Add timestamps to EOPatch
        eopatch.timestamp = transformed_timestamps

        # Load and add mask features from TIFF
        LOGGER.info(f'Loading mask data from {mask_tiff_path}')
        with rasterio.open(mask_tiff_path) as src:
            mask_data = src.read()

        # Check the dimensions of the mask data
        if mask_data.shape[0] != 4:
            raise ValueError("The mask TIFF should have 4 bands: extent, boundary, distance, enum")

        # Extract each band
        extent = mask_data[0]
        boundary = mask_data[1]
        distance = mask_data[2]
        enum = mask_data[3]

        # Add each mask as a feature in the EOPatch
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'EXTENT', extent[..., np.newaxis])
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'BOUNDARY', boundary[..., np.newaxis])
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'DISTANCE', distance[..., np.newaxis])
        eopatch.add_feature(FeatureType.MASK_TIMELESS, 'ENUM', enum[..., np.newaxis])

        # Verify that the features have been added
        if 'EXTENT' not in eopatch.mask_timeless:
            LOGGER.error("Failed to add EXTENT mask to EOPatch")
        if 'BOUNDARY' not in eopatch.mask_timeless:
            LOGGER.error("Failed to add BOUNDARY mask to EOPatch")
        if 'DISTANCE' not in eopatch.mask_timeless:
            LOGGER.error("Failed to add DISTANCE mask to EOPatch")
        if 'ENUM' not in eopatch.mask_timeless:
            LOGGER.error("Failed to add ENUM mask to EOPatch")

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
    mask_tiff_path = '/home/joseph/Code/localstorage/dataset/sentinel2/masks/FR/FR_9334_S2label_10m_256.tif'
    output_eopatch_dir = '/home/joseph/Code/localstorage/eopatchs/'

    # Create EOPatch from NetCDF and TIFF data
    create_eopatch_from_nc_and_tiff(nc_file_path, vector_data_path, mask_tiff_path, output_eopatch_dir)

if __name__ == "__main__":
    main()
