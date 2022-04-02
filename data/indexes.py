from os import path
from os import listdir

SUPPORTED_DATASETS = [
    "kitti2012",
    "kitti2015",
    "kittis",
    "driving",
    "flyingthings3d",
    "monkaa",
    "sceneflow",
]

def check_paths_exist(*args):
    for a in args:
        if not path.exists(a):
            raise ValueError(f"path {a} does not exist")

def match_images_disparities(left_folder, right_folder, disp_folder, input_extension):
    """
    Matches image pairs with disparity files into tuples of three
    """
    check_paths_exist(left_folder,right_folder,disp_folder)
    triplets = []
    for disparity_file in listdir(disp_folder):
        filename = path.splitext(disparity_file)[0]
        image_name = f"{filename}.{input_extension}"

        disp_file = path.join(disp_folder, disparity_file)
        left_image = path.join(left_folder, image_name)
        right_image = path.join(right_folder, image_name)
        if (
            path.isfile(disp_file)
            and path.isfile(left_image)
            and path.isfile(right_image)
        ):
            triplets.append((left_image, right_image, disp_file))
    return triplets


def index_kitti2012(root, occlussion=True, split=0.8, colored=True, **kwargs):
    disp_folder = "disp_occ"
    if not occlussion:
        disp_folder = "disp_noc"
    if colored:
        return __index_kitti(root, "colored_0", "colored_1", disp_folder)
    return __index_kitti(root, "image_0", "image_1", disp_folder, split=split)


def index_kitti2015(root, occlussion=True, split=0.8, **kwargs):
    disp_folder = "disp_occ_0"
    if not occlussion:
        disp_folder = "disp_noc_0"
    return __index_kitti(root, "image_2", "image_3", disp_folder, split=split)


def __index_kitti(
    root, left_folder, right_folder, disp_folder, input_extension="png", split=0.8
):
    if split < 0 or split > 1:
        raise ValueError("split should be a float between 0 and 1")
    left = path.join(root, "training", left_folder)
    right = path.join(root, "training", right_folder)
    disparity = path.join(root, "training", disp_folder)
    data = match_images_disparities(left, right, disparity, input_extension)
    return data[: int(len(data) * split)], data[int(len(data) * split) :]


def combine_kitti(root, occlussion=True, split=0.8, **kwargs):
    kitti2012_folder = path.join(root, "data_stereo_flow")
    kitti2015_folder = path.join(root, "data_scene_flow")
    kitti2012, kitti2012_test = index_kitti2012(kitti2012_folder, occlussion, split)
    kitti2015, kitti2015_test = index_kitti2015(kitti2015_folder, occlussion, split)
    trainset = kitti2012 + kitti2015
    testset = kitti2012_test + kitti2015_test
    return trainset, testset


def index_driving(
    root_images, root_disparity, webp=True, disparity_side="left", split=0.8, **kwargs
):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    if split < 0 or split > 1:
        raise ValueError("split should be a float between 0 and 1")
    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not webp:
        maindir = "frames_cleanpass"
        extension = "png"

    internal_paths = (
        path.join("15mm_focallength", "scene_backwards", "fast"),
        path.join("15mm_focallength", "scene_backwards", "slow"),
        path.join("15mm_focallength", "scene_forwards", "fast"),
        path.join("15mm_focallength", "scene_forwards", "slow"),
        path.join("35mm_focallength", "scene_backwards", "fast"),
        path.join("35mm_focallength", "scene_backwards", "slow"),
        path.join("35mm_focallength", "scene_forwards", "fast"),
        path.join("35mm_focallength", "scene_forwards", "slow"),
    )

    data = []
    for p in internal_paths:
        left = path.join(root_images, maindir, p, "left")
        right = path.join(root_images, maindir, p, "right")
        disparity = path.join(root_disparity, "disparity", p, disparity_side)

        triplets = match_images_disparities(left, right, disparity, extension)
        data.extend(triplets)

    return data[: int(len(data) * split)], data[int(len(data) * split) :]


def index_flyingthings(
    root_images, root_disparity, webp=True, disparity_side="left", split=0.8, **kwargs
):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    if split < 0 or split > 1:
        raise ValueError("split should be a float between 0 and 1")
    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not webp:
        maindir = "frames_cleanpass"
        extension = "png"

    def __index_flyingthings(subfolder1):
        data = []
        for subfolder2 in ["A", "B", "C"]:
            folder = path.join(root_disparity, "disparity", subfolder1, subfolder2)
            subfolders3 = [path.join(folder, sf) for sf in listdir(folder)]
            subfolders3 = [f for f in subfolders3 if path.isdir(f)]
            for sf in listdir(folder):
                img_path = path.join(root_images, maindir, subfolder1, subfolder2, sf)
                left = path.join(img_path, "left")
                right = path.join(img_path, "right")
                disparity = path.join(folder, sf, disparity_side)
                triplets = match_images_disparities(left, right, disparity, extension)
                data.extend(triplets)
        return data

    train_data = __index_flyingthings("TRAIN")
    test_data = __index_flyingthings("TEST")

    return train_data, test_data


def index_monkaa(
    root_images, root_disparity, webp=True, disparity_side="left", split=0.8, **kwargs
):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    if split < 0 or split > 1:
        raise ValueError("split should be a float between 0 and 1")

    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not webp:
        maindir = "frames_cleanpass"
        extension = "png"

    data = []
    for subfolder in listdir(path.join(root_images, maindir)):
        left = path.join(root_images, maindir, subfolder, "left")
        right = path.join(root_images, maindir, subfolder, "right")
        disparity = path.join(root_disparity, "disparity", subfolder, disparity_side)
        data.extend(match_images_disparities(left, right, disparity, extension))
    return data[: int(len(data) * split)], data[int(len(data) * split) :]


def combine_sceneflow(root, webp=True, disparity_side="left", split=0.8, **kwargs):
    if root is None:
        raise ValueError("root folder is not set")
    if not path.exists(root):
        raise ValueError(f"path {root} does not exist")
    if webp:
        monkaa_images = path.join(root, "monkaa__frames_cleanpass_webp")
        driving_images = path.join(root, "driving__frames_cleanpass_webp")
        flying_images = path.join(root, "flyingthings3d__frames_cleanpass_webp")
    else:
        monkaa_images = path.join(root, "monkaa__frames_cleanpass")
        driving_images = path.join(root, "driving__frames_cleanpass")
        flying_images = path.join(root, "flyingthings3d__frames_cleanpass")
    monkaa_disparity = path.join(root, "monkaa__disparity")
    driving_disparity = path.join(root, "driving__disparity")
    flying_disparity = path.join(root, "flyingthings3d__disparity")

    check_paths_exist(monkaa_images,monkaa_disparity,driving_images,driving_disparity,flying_images,flying_disparity)

    driving, driving_test = index_driving(
        driving_images, driving_disparity, webp, disparity_side, split
    )
    flying, flying_test = index_flyingthings(
        flying_images, flying_disparity, webp, disparity_side, split
    )
    monkaa, monkaa_test = index_monkaa(
        monkaa_images, monkaa_disparity, webp, disparity_side, split
    )

    trainset = driving + flying + monkaa
    testset = driving_test + flying_test + monkaa_test

    return trainset, testset


def index_set(dataset_name, **kwargs):
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


def main():
    a, aa = index_set(
        "kitti2012",
        root="E:\\Thesis\\Datasets\\data_stereo_flow\\training",
        occlussion=True,
        colored=True,
        split=0.8,
    )
    b, bb = index_set(
        "kitti2015",
        root="E:\\Thesis\\Datasets\\data_scene_flow\\training",
        occclussion=True,
        split=0.8,
    )
    c, cc = index_set(
        "kittis",
        root="E:\\Thesis\\Datasets",
        occlussion=True,
        split=0.8,
    )
    d, dd = index_set(
        "driving",
        root_images="D:\\thesis_data\\driving__frames_cleanpass_webp",
        root_disparity="D:\\thesis_data\\driving__disparity",
        webp=True,
        disparity_side="left",
        split=0.8,
    )
    e, ee = index_set(
        "flyingthings3d",
        root_images="D:\\thesis_data\\flyingthings3d__frames_cleanpass_webp",
        root_disparity="D:\\thesis_data\\flyingthings3d__disparity",
        webp=True,
        disparity_side="left",
        split=0.8,
    )
    f, ff = index_set(
        "monkaa",
        root_images="D:\\thesis_data\\monkaa__frames_cleanpass_webp",
        root_disparity="D:\\thesis_data\\monkaa__disparity",
        webp=True,
        disparity_side="left",
        split=0.8,
    )
    g, gg = index_set(
        "sceneflow",
        root="D:\\thesis_data\\",
        webp=True,
        disparity_side="left",
        split=0.8,
    )
    for x in [a, b, c, d, e, f, g]:
        print(len(x))


def main1():
    kittis, kittis_t = combine_kitti("E:\\Thesis\\Datasets")
    sceneflow, sceneflow_t = combine_sceneflow("D:\\thesis_data\\")
    data1_tr, data1_te = index_kitti2012(
        "E:\\Thesis\\Datasets\\data_stereo_flow\\training"
    )

    data2_tr, data2_te = index_kitti2015(
        "E:\\Thesis\\Datasets\\data_scene_flow\\training"
    )

    data3_tr, data3_te = index_driving(
        "D:\\thesis_data\\driving__frames_cleanpass_webp",
        "D:\\thesis_data\\driving__disparity",
    )

    data4_tr, data4_te = index_flyingthings(
        "D:\\thesis_data\\flyingthings3d__frames_cleanpass_webp",
        "D:\\thesis_data\\flyingthings3d__disparity",
    )

    data5_tr, data5_te = index_monkaa(
        "D:\\thesis_data\\monkaa__frames_cleanpass_webp",
        "D:\\thesis_data\\monkaa__disparity",
    )

    print(len(kittis), len(kittis_t))
    print(len(sceneflow), len(sceneflow_t))
    print(len(data1_tr), len(data2_tr), len(data3_tr), len(data4_tr), len(data5_tr))
    print(len(data1_te), len(data2_te), len(data3_te), len(data4_te), len(data5_te))


def test(root):
    print(root)


if __name__ == "__main__":
    # main()
    # main1()
    test(**{"root": "this is root lmao"})
