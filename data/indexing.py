from .indexing_drivingstereo import combine_weathers, index_weather
from .indexing_kitty import combine_kitti, index_kitti2012, index_kitti2015
from .indexing_sceneflow import (combine_sceneflow, index_driving,
                                 index_flyingthings, index_monkaa)

SUPPORTED_DATASETS = [
    "kitti2012",
    "kitti2015",
    "kittis",
    "driving",
    "flyingthings3d",
    "monkaa",
    "sceneflow",
    "ds-weather",
    "ds-all-weather",
]


def index_set(dataset_name, **kwargs):
    """
    Passes arguments to correct indexing function depending on the name
    """
    if (
        dataset_name in ["sceneflow", "kittis"]
        and kwargs.get("validation_length", -1) != -1
    ):
        raise ValueError(
            "Combined datasets sceneflow and kittis do not support validation_length"
        )
    if not dataset_name or len(dataset_name) == 0:
        raise ValueError("dataset name not specified")
    if not kwargs.get("root"):
        raise KeyError("Root path for dataset not defined")

    indexers = {
        "kitti2012": index_kitti2012,
        "kitti2015": index_kitti2015,
        "kittis": combine_kitti,
        "driving": index_driving,
        "flyingthings3d": index_flyingthings,
        "monkaa": index_monkaa,
        "sceneflow": combine_sceneflow,
        "ds-weather": index_weather,
        "ds-all-weather": combine_weathers,
    }
    index = indexers.get(dataset_name)
    if index == None:
        raise KeyError("dataset of given name not found")
    return index(**kwargs)
