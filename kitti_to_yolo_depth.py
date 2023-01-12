"""
generate yolo format labels (including depth as the last column) from kitti format labels
"""

import os
import shutil
import socket
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 0


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


cls_names = [
    "Pedestrian",
    "Car",
    "Van",
    "Truck",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
    "DontCare",
]

classes = ["Person", "Vehicle"]

classes_dict = {
    "Pedestrian": 0,
    "Person_sitting": 0,
    "Cyclist": 0,
    "Car": 1,
    "Van": 1,
    "Truck": 1,
    "Tram": 1,
}


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


def txt_kitti2yolo(image_path_list, label_path_list):
    """
    Args:
        image_path: path list to images
        label_path: path list to labels
        xyxy: True if label is in xyxy format, False if label is in xywh format

    Return:
        txt_list: LIST of list of strings,
                    each string is A line in txt file
                    each item in list is ALL lines of txt file
                    each item in LIST is a list of strings extracted form ONE txt file
    """
    txt_list = []

    for image_path, label_path in zip(image_path_list, label_path_list):
        with open(label_path) as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]

            # # if not in classes, skip
            # if not any([line[0] in classes for line in lines]):
            #     continue

            # to store labels for one file
            txt_temp = []

            # read image info
            img = cv2.imread(str(image_path))
            height, width, channels = img.shape

            # deal with each line in label file
            for line in lines:
                if (
                    line[0] == "DontCare" or line[0] == "Misc"
                ):  # TODO: Is this the best way to deal with 'DontCare' and 'Misc'?
                    continue
                # kitti originally records top-left and bottom-right coordinates
                # but coco records center and size
                c = int(classes_dict[line[0]])
                x0 = float(line[4]) / width
                y0 = float(line[5]) / height
                x1 = float(line[6]) / width
                y1 = float(line[7]) / height

                # preprocessing for depth
                # TODO: is it the best way?
                # clip negative depth to zero, so that depths are in [0, 146.85]
                d = max(0, float(line[13]))
                # change to [0, 1] by dividing 146.85
                # d /= 146.85

                # xyxy to xywh
                w = x1 - x0
                h = y1 - y0
                xc = (x0 + x1) / 2
                yc = (y0 + y1) / 2

                # keep 6 digits as in coco
                w = round(w, 6)
                h = round(h, 6)
                xc = round(xc, 6)
                yc = round(yc, 6)
                d = round(d, 6)

                # append to txt_temp
                txt_temp.append(f"{c} {xc} {yc} {w} {h} {d}")
            if len(txt_temp) == []:
                print(f"no interesting label in {label_path}")
            # save new txt
            txt_list.append(txt_temp)
    return txt_list


def copy_paste_image(path_list, dst):
    """
    Copies files from path_list to dst.
    Args:
      path_list list: list of paths to copy
      dst pathlib.Path: destination folder
    """
    if dst.is_dir():
        print(f"Directory {dst} already exists.")
        print("Nothing done.")
    else:
        # print(f"Did not find {dst} directory.")
        dst.mkdir(parents=True, exist_ok=True)
        print(f"Directory {dst} created.")
        print(f"Copying files to {dst}...")
        for file in path_list:
            shutil.copy(file, dst)
        print("Done.")


def move_image(path_list, dst):
    """
    Copies files from path_list to dst.
    Args:
      path_list list: list of paths to move
      dst pathlib.Path: destination folder
    """
    if dst.is_dir():
        print(f"Directory {dst} already exists.")
        print("Nothing done.")
    else:
        # print(f"Did not find {dst} directory.")
        dst.mkdir(parents=True, exist_ok=True)
        print(f"Directory {dst} created.")
        print(f"Moving files to {dst}...")
        for file in path_list:
            shutil.move(file, dst)
        print("Done.")


def copy_paste_label(path_list, txt_list, dst, depth=False):
    """
    Copies files from path_list, to dst.
    Args:
      path_list list: list of paths of source label
      txt_list list: list of actual labels to copy
      dst pathlib.Path: destination folder
      depth bool: True if depth is included in label
    """
    if dst.is_dir():
        print(f"Directory {dst} already exists.")
        print("Nothing done.")
    else:
        # print(f"Did not find {dst} directory.")
        dst.mkdir(parents=True, exist_ok=True)
        print(f"Directory {dst} created.")
        print(f"Copying files to {dst}...")

        for i, file in enumerate(path_list):
            # create new label file
            label_file = dst / file.name
            with open(label_file, "w") as f:
                for line in txt_list[i]:
                    if depth:
                        f.write(line + "\n")
                    else: # remove depth
                        f.write(' '.join(line.split()[:-1]) + "\n")
        print("Done.")



if __name__ == "__main__":

    colors = Colors()

    hostname = socket.gethostname()
    if hostname == "HAITI":
        # repo_folder = "C:/Users/xin/OneDrive - bwstaff/xin/yolov5/"
        dataset_folder = "D:/xin/datasets/CV/kitti/"
        # dataset_folder = "C:/Users/xin/Downloads/kitti"
    if hostname == "BALI":
        # repo_folder = "/home//OneDrive/xin/yolov5/"
        dataset_folder = "/storage/xin/datasets/CV/kitti/"
        dataset_folder = "/home/xin/Datasets/kitti"
    if hostname == "DESKTOP-SJ7KPPD":
        # repo_folder = "C:/Users/Xin/OneDrive - bwstaff/xin/yolov5/"
        # repo_folder = "C:/Users/Xin/git_repo_local/yolov5_local/"
        dataset_folder = "C:/Users/Xin/Datasets/kitti"
    if hostname == "chen-Ubuntu":
        # repo_folder = "C:/Users/Xin/OneDrive - bwstaff/xin/yolov5/"
        # repo_folder = "C:/Users/Xin/git_repo_local/yolov5_local/"
        dataset_folder = ".."

    # repo_folder = Path(repo_folder)
    dataset_folder = Path(dataset_folder)
    dataset_source_folder = (
        dataset_folder / "Source"
    )  # original downloaded dataset is here

    # source image/label paths
    src_training_folder = dataset_source_folder / "training"
    src_training_image_folder = src_training_folder / "image_2"
    src_training_image_path_list = sorted(
        list(src_training_image_folder.glob("**/*.png"))
    )
    src_training_label_folder = src_training_folder / "label_2"
    src_training_label_path_list = sorted(
        list(src_training_label_folder.glob("**/*.txt"))
    )

    # train val split # TODO: find an official split
    (
        kitti_training_image_list,
        kitti_val_image_list,
        kitti_training_label_list,
        kitti_val_label_list,
    ) = train_test_split(
        src_training_image_path_list,
        src_training_label_path_list,
        test_size=0.2,
        random_state=SEED,
    )

    # generate txt_list
    print("Generating yolo format txt lists for training datasets...")
    train_txt_list = txt_kitti2yolo(
        kitti_training_image_list, kitti_training_label_list
    )
    print("Generating yolo format txt lists for val datasets...")
    val_txt_list = txt_kitti2yolo(kitti_val_image_list, kitti_val_label_list)

    # copy paste images and labels (WITHOUT depth) to new folder
    kitti_folder = dataset_folder / "kitti"
    image_dir = kitti_folder / "images"
    label_dir = kitti_folder / "labels"
    copy_paste_image(kitti_training_image_list, image_dir / "train")
    copy_paste_image(kitti_val_image_list, image_dir / "val")
    copy_paste_label(kitti_val_label_list, val_txt_list, label_dir / "train", depth=False)
    copy_paste_label(kitti_val_label_list, val_txt_list, label_dir / "val", depth=False)

    # copy paste images and labels (WITH depth) to new folder
    kitti_folder_depth = dataset_folder / "kitti_depth"
    image_dir_depth = kitti_folder_depth / "images"
    label_dir_depth = kitti_folder_depth / "labels"
    copy_paste_image(kitti_training_image_list, image_dir_depth / "train")
    copy_paste_image(kitti_val_image_list, image_dir_depth / "val")
    copy_paste_label(kitti_training_label_list, train_txt_list, label_dir_depth / "train", depth=True)
    copy_paste_label(kitti_val_label_list, val_txt_list, label_dir_depth / "val", depth=True)

