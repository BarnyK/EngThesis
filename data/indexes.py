from os import path
from os import listdir

def match_images_disparities(left_folder, right_folder, disp_folder, input_extension):
    """
    Matches image pairs with disparity files into tuples of three
    """
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


def index_kitti2012(root, colored=True, occ=True,split=0.8):
    disp_folder = "disp_occ"
    if not occ:
        disp_folder = "disp_noc"
    if colored:
        return index_kitti(root, "colored_0", "colored_1", disp_folder)
    return index_kitti(root, "image_0", "image_1", disp_folder,split=split)


def index_kitti2015(root, occ=True,split=0.8):
    disp_folder = "disp_occ_0"
    if not occ:
        disp_folder = "disp_noc_0"
    return index_kitti(root, "image_2", "image_3", disp_folder,split=split)


def index_kitti(root, left_folder, right_folder, disp_folder, input_extension="png",split=0.8):
    left = path.join(root, left_folder)
    right = path.join(root, right_folder)
    disparity = path.join(root, disp_folder)
    data = match_images_disparities(left, right, disparity, input_extension)
    return data[:int(len(data)*split)], data[int(len(data)*split):]



def index_driving(root_images, root_disparity, webp=True, disparity_side="left",split=0.8):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not maindir:
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

    return data[:int(len(data)*split)], data[int(len(data)*split):]


def index_flyingthings(root_images, root_disparity,disparity_side="left",split=0.8):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")

    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not maindir:
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


def index_monkee(root_images, root_disparity, webp=True, disparity_side="left",split=0.8):
    if disparity_side not in ("left", "right"):
        raise ValueError("disparity_side should be either 'left' or 'right'")
    maindir = "frames_cleanpass_webp"
    extension = "webp"
    if not maindir:
        maindir = "frames_cleanpass"
        extension = "png"

    data = []
    for subfolder in listdir(path.join(root_images, maindir)):
        left = path.join(root_images, maindir, subfolder, "left")
        right = path.join(root_images, maindir, subfolder, "right")
        disparity = path.join(root_disparity, "disparity", subfolder, disparity_side)
        data.extend(match_images_disparities(left, right, disparity, extension))
    return data[:int(len(data)*split)], data[int(len(data)*split):]


def main():
    data1_tr,data1_te = index_kitti2012(
        "E:\\Thesis\\Datasets\\Kitty2012\\data_stereo_flow\\training"
    )

    data2_tr,data2_te = index_kitti2015(
        "E:\\Thesis\\Datasets\\Kitty2015\\data_scene_flow\\training"
    )

    data3_tr,data3_te = index_driving(
        "D:\\thesis_data\\driving__frames_cleanpass_webp",
        "D:\\thesis_data\\driving__disparity",
    )

    data4_tr,data4_te = index_flyingthings(
        "D:\\thesis_data\\flyingthings3d__frames_cleanpass_webp",
        "D:\\thesis_data\\flyingthings3d__disparity",
    )

    data5_tr,data5_te = index_monkee(
        "D:\\thesis_data\\monkaa__frames_cleanpass_webp",
        "D:\\thesis_data\\monkaa__disparity",
    )

    print(len(data1_tr), len(data2_tr), len(data3_tr), len(data4_tr), len(data5_tr))
    print(len(data1_te), len(data2_te), len(data3_te), len(data4_te), len(data5_te))


if __name__ == "__main__":
    main()
