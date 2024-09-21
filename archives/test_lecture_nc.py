import xarray as xr

def print_metadata(ds):
    """
    Print metadata of the NetCDF file using xarray.

    Parameters:
    ds (xarray.Dataset): The xarray dataset object.
    """
    print("Meta data:")
    print(f"Dimensions: {list(ds.dims.keys())}")
    print(f"Variables: {list(ds.variables.keys())}")
    
    print("\nGlobal Attributes:")
    for attr in ds.attrs:
        print(f"  {attr}: {ds.attrs[attr]}")

    print("\nVariable Information:")
    for var in ds.variables:
        print(f"\nVariable: {var}")
        print(f"  Dimensions: {ds[var].dims}")
        print(f"  Size: {ds[var].size}")
        print(f"  Attributes:")
        for attr in ds[var].attrs:
            print(f"    {attr}: {ds[var].attrs[attr]}")

def print_data(ds):
    """
    Print data of the NetCDF file using xarray.

    Parameters:
    ds (xarray.Dataset): The xarray dataset object.
    """
    for var in ds.variables:
        data = ds[var].values
        print(f"\nData for variable '{var}':")
        print(data)

def main():
    # Define path to the NetCDF file
    nc_file_path = '/home/joseph/Code/localstorage/dataset/sentinel2/images/FR/FR_9334_S2_10m_256.nc'

    # Open the NetCDF file using xarray
    ds = xr.open_dataset(nc_file_path)

    # Print metadata
    print_metadata(ds)
    
    # Print data
    print_data(ds)

if __name__ == "__main__":
    main()
