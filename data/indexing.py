from os import path
from os import listdir
from .indexing_kitty import index_kitti2012, index_kitti2015, combine_kitti
from .utils import check_paths_exist
from .indexing_sceneflow import (
    index_driving,
    index_flyingthings,
    index_monkaa,
    combine_sceneflow,
)

SUPPORTED_DATASETS = [
    "kitti2012",
    "kitti2015",
    "kittis",
    "driving",
    "flyingthings3d",
    "monkaa",
    "sceneflow",
]


def index_set(dataset_name, **kwargs):
    """
    Passes arguments to correct indexing function depending on the name
    """
    indexers = {
        "kitti2012": index_kitti2012,
        "kitti2015": index_kitti2015,
        "kittis": combine_kitti,
        "driving": index_driving,
        "flyingthings3d": index_flyingthings,
        "monkaa": index_monkaa,
        "sceneflow": combine_sceneflow,
    }
    index = indexers.get(dataset_name)
    if index == None:
        raise KeyError("dataset of given name not found")
    return index(**kwargs)
