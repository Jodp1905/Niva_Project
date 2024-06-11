# Test sans ImportFromTiff
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Union
import rasterio
import geopandas as gpd
import fs
import numpy as np
from eolearn.core import EOPatch, EOTask, EOWorkflow, SaveTask, MergeFeatureTask, RemoveFeature, RenameFeature, OverwritePermission, LinearWorkflow, FeatureType
from eolearn.core.fs_utils import get_base_filesystem_and_path
from s2cloudless import S2PixelCloudDetector
from sentinelhub import SHConfig, CRS, Geometry
from .utils import set_sh_config, BaseConfig


@dataclass
class TiffsToEopatchConfig(BaseConfig):
    tiffs_folder: str
    eopatches_folder: str
    band_names: List[str]
    mask_name: str
    data_name: str = 'BANDS'
    is_data_mask: str = 'IS_DATA'
    clp_name: str = 'CLP'
    clm_name: str = 'CLM'


class AddTimestampsUpdateTime(EOTask):
    def __init__(self, path: str):
        self.path = path

    def _get_valid_dates(self, tile_name: str, filename: str = 'userdata.json') -> List[datetime]:
        filesystem, relative_path = get_base_filesystem_and_path(self.path)
        full_path = fs.path.join(relative_path, tile_name, filename)
        decoded_data = filesystem.readtext(full_path, encoding='utf-8')
        parsed_data = json.loads(decoded_data)
        dates = json.loads(parsed_data['dates'])
        return [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    def execute(self, eopatch: EOPatch, *, tile_name: str) -> EOPatch:
        dates = self._get_valid_dates(tile_name)
        eopatch.timestamp = dates
        return eopatch


class RearrangeBands(EOTask):
    def execute(self, eopatch: EOPatch) -> EOPatch:
        for band in eopatch.data:
            eopatch.data[band] = np.swapaxes(eopatch.data[band], 0, -1)
        for mask in eopatch.mask:
            eopatch.mask[mask] = np.swapaxes(eopatch.mask[mask], 0, -1)
        return eopatch


class DeleteFiles(EOTask):
    def __init__(self, path: str, filenames: List[str]):
        self.path = path
        self.filenames = filenames

    def execute(self, eopatch: Union[EOPatch, str], *, tile_name: str):
        filesystem, relative_path = get_base_filesystem_and_path(self.path)
        for filename in self.filenames:
            full_path = fs.path.join(relative_path, tile_name, filename)
            filesystem.remove(full_path)


class CloudMasking(EOTask):
    def __init__(self, clp_feature: Tuple = (FeatureType.DATA, 'CLP'), clm_feature: Tuple = (FeatureType.MASK, 'CLM'),
                 average_over: int = 24, max_clp: float = 255.):
        self.clm_feature = next(self._parse_features(clm_feature)())
        self.clp_feature = next(self._parse_features(clp_feature)())
        self.s2_cd = S2PixelCloudDetector(average_over=average_over)
        self.max_clp = max_clp

    def execute(self, eopatch: EOPatch) -> EOPatch:
        clc = self.s2_cd.get_mask_from_prob(eopatch[self.clp_feature].squeeze() / self.max_clp)
        eopatch[self.clm_feature] = clc[..., np.newaxis]
        return eopatch


class ImportTiffWithRasterio(EOTask):
    def __init__(self, feature: Tuple[FeatureType, str], path: str):
        self.feature = feature
        self.path = path

    def execute(self, eopatch: EOPatch, *, tile_name: str):
        full_path = f"{self.path}/{tile_name}/{self.feature[1]}.tif"
        with rasterio.open(full_path) as src:
            data = src.read()
        eopatch[self.feature] = np.moveaxis(data, 0, -1)[..., np.newaxis]
        return eopatch


def convert_tiff_to_eopatches(config: TiffsToEopatchConfig, delete_tiffs: bool = False):
    sh_config = set_sh_config(config)
    
    import_bands = [(ImportTiffWithRasterio((FeatureType.DATA, band), config.tiffs_folder), f'Import band {band}')for band in config.band_names]

    import_clp = (ImportTiffWithRasterio((FeatureType.DATA, config.clp_name), config.tiffs_folder), f'Import {config.clp_name}')
    
    import_mask = (ImportTiffWithRasterio((FeatureType.MASK, config.mask_name), config.tiffs_folder), f'Import {config.mask_name}')

    rearrange_bands = (RearrangeBands(), 'Swap time and band axis')
    
    add_timestamps = (AddTimestampsUpdateTime(config.tiffs_folder), 'Load timestamps')

    merge_bands = (MergeFeatureTask(
        input_features={FeatureType.DATA: config.band_names},
        output_feature=(FeatureType.DATA, config.data_name)), 'Merge band features')

    remove_bands = (RemoveFeature(features={FeatureType.DATA: config.band_names}), 'Remove bands')

    rename_mask = (RenameFeature((FeatureType.MASK, config.mask_name, config.is_data_mask)), 'Rename is data mask')

    calculate_clm = (CloudMasking(), 'Get CLM mask from CLP')

    save_task = (SaveTask(path=config.eopatches_folder, config=sh_config,
                          overwrite_permission=OverwritePermission.OVERWRITE_FEATURES),  'Save EOPatch')

    filenames = [f'{band}.tif' for band in config.band_names] + \
                [f'{config.mask_name}.tif', f'{config.clp_name}.tif', 'userdata.json']
    delete_files = (DeleteFiles(path=config.tiffs_folder, filenames=filenames), 'Delete batch files')

    workflow = [*import_bands,
                import_clp,
                import_mask,
                rearrange_bands,
                add_timestamps,
                merge_bands,
                remove_bands,
                rename_mask,
                calculate_clm,
                save_task]

    if delete_tiffs:
        workflow.append(delete_files)

    eo_workflow = LinearWorkflow(*workflow)
    exec_args = get_exec_args(eo_workflow, [name for name in config.band_names])

    results = eo_workflow.execute(exec_args)
    return results
#qwerty
#qwerty
#qwerty
#qwerty
#qwerty
#qwerty
#qwerty
#qwerty
#qwerty

def get_exec_args(workflow: EOWorkflow, eopatch_list: List[str]) -> List[dict]:
    exec_args = []
    print('hello')
    tasks = workflow.get_tasks()

    for name in eopatch_list:
        single_exec_dict = {}

        for task_name, task in tasks.items():
            if isinstance(task, ImportTiffWithRasterio):
                tiff_name = task_name.split()[-1]
                path = f'{name}/{tiff_name}.tif'
                single_exec_dict[task] = dict(tile_name=path)

            if isinstance(task, SaveTask):
                single_exec_dict[task] = dict(eopatch_folder=name)

            if isinstance(task, (AddTimestampsUpdateTime, DeleteFiles)):
                single_exec_dict[task] = dict(tile_name=name)

        exec_args.append(single_exec_dict)

    return exec_args
