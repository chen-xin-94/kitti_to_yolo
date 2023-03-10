import os
import shutil
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# original class names
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
cls_dict = {n: i for i, n in enumerate(cls_names)}

# modified class names for data extraction
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
# special dict for plot_box_depth()
classes_dict_inv = {i: c for i, c in enumerate(classes)}

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


colors = Colors()


def observe(
    num,
    image_list,
    label_list,
    xyxy=True,
):
    image = image_list[num]
    label = label_list[num]
    print(image)
    print(label)
    img = cv2.imread(str(image))
    height, width, channels = img.shape
    print(height, width, channels)
    with open(label) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        print(line)
        if xyxy:
            # kitti originally records top-left and bottom-right coordinates
            n = int(cls_dict[line[0]])
            xyxy = [int(float(x)) for x in line[4:8]]  # TODO: ?????????bounding box?????????
            cv2.rectangle(
                img,
                tuple(xyxy[0:2]),
                tuple(xyxy[2:4]),
                color=colors(n, True),
                thickness=5,
            )
        else:
            # xywh and in yolo format
            n = int(line[0])
            xc = int(width * float(line[1]))
            yc = int(height * float(line[2]))
            w = int(width * float(line[3]))
            h = int(height * float(line[4]))
            cv2.rectangle(
                img,
                (xc - w // 2, yc - h // 2),
                (xc + w // 2, yc + h // 2),
                color=colors(n, True),
                thickness=5,
            )

    plt.figure(figsize=(int(10 * width / height), 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


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
                    else:  # remove depth
                        f.write(" ".join(line.split()[:-1]) + "\n")
        print("Done.")

def box_label(im, box, label, color=(255, 0, 0), txt_color=(255, 255, 255)):
    lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            im,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def plot_box_depth(num, image_list, label_list):
    img = cv2.imread(str(image_list[num]))
    h, w, _ = img.shape
    label_path = label_list[num]
    with open(label_path) as f:
        targets = f.readlines()
    targets = [line.split() for line in targets]
    # gt in format [cls, x, y, w, h, depth]
    for target in targets:
        print(target)
    for target in targets:
        # extract box in xywh
        box = [float(x) for x in target[1:5]]
        # convert to xyxy
        box = xywh2xyxy(np.array(box)[None])[0]
        # convert to pixel
        box = box * np.array([w, h, w, h])
        # extract depth
        depth = float(target[5])
        # extract label
        c = int(target[0])
        color = colors(c)
        label = classes_dict_inv[c]
        # # add fake conf
        # label = label + " " + "0.9"
        # add depth
        # d = round(float(target[-1])*146.85,2)
        d = round(float(target[-1]), 2)
        label = label + " " + str(d) + "m"
        # draw for this line
        box_label(img, box, label, color)
    plt.figure(figsize=(int(10 * w / h), 10))
    plt.imshow(img)