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

from utils import Colors, walk_through_dir, txt_kitti2yolo, copy_paste_image, move_image, copy_paste_label
from utils import classes_dict, classes, cls_names

SEED = 0

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

    # copy paste images and labels to new folder
    kitti_folder = dataset_folder / "kitti"
    image_dir = kitti_folder / "images"
    label_dir = kitti_folder / "labels"
    label_dir_depth = kitti_folder / "labels_depth"

    # copy paste images
    copy_paste_image(kitti_training_image_list, image_dir / "train")
    copy_paste_image(kitti_val_image_list, image_dir / "val")

    # labels without depth
    copy_paste_label(kitti_training_label_list, train_txt_list, label_dir / "train", depth=False)
    copy_paste_label(kitti_val_label_list, val_txt_list, label_dir / "val", depth=False)

    # labels with depth
    copy_paste_label(kitti_training_label_list, train_txt_list, label_dir_depth / "train", depth=True)
    copy_paste_label(kitti_val_label_list, val_txt_list, label_dir_depth / "val", depth=True)

