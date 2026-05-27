import os
import cv2
import numpy as np
import csv

def find_image(name, root_dir):
    for root, dirs, files in os.walk(root_dir):
        if name in files:
            # return os.path.join(root, name)
            return root

    return None

def load_image(name, image_dir):
    image_path = os.path.join(image_dir, name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return image

def load_std_depth(name, dir):
    try:
        base_name = os.path.splitext(name)[0]
        depth_path = os.path.join(dir, f"{base_name}.npy")
        depth = np.load(depth_path).astype(np.float32)
        return depth
    except:
        return None

def load_depth(name, depth_dir, depth_type):

    name = os.path.splitext(name)[0]
    
    if depth_type == "MARIGOLD":
        # construct the depth name
        depth_name = f"{name}_depth.npy"
    elif depth_type == "DEPTH_PRO":
        depth_name = f"{name}.npz"

    else:
        raise RuntimeError(f"Depth type: {depth_type} no supported")

    depth_path = os.path.join(depth_dir, depth_name)


    if not os.path.exists(depth_path):
        raise RuntimeError(f"Could not find depth corresponding depth file: {depth_path}")

    if depth_type == "MARIGOLD":
        depth = np.load(depth_path).astype(np.float32)
    elif depth_type == "DEPTH_PRO":
        depth = np.load(depth_path)
        depth = depth["depth"].astype(np.float32)
    else:
        raise RuntimeError(f"Depth type: {depth_type} no supported")

    return depth

# def load_eval_scores(name, database):
#     # open the database
#     
#     with open(database, "r", newline="") as f:
#         reader = csv.DictReader(f)
#
#         for row in reader:
#             if row["image_path"] == name:
#                 savoyness = row["savoyness"]
#                 cupping = row["cupping"]
#
#                 savoyness = int(savoyness) if savoyness != "" else None
#                 cupping = int(cupping) if cupping != "" else None
#
#                 return savoyness, cupping
#
#     return None, None

def is_numeric(string):
    return string.lstrip("-+").isdigit()

def load_plot_eval_scores(plot_num, database_file):
    with open(database_file, "r", newline="") as f:
        reader=csv.DictReader(f)

        for row in reader:
            if int(row["PLOT"]) == plot_num:
                savoyness = row["LF_SAVOY"]
                cupping = row["LF_CUP"]

                savoyness = int(savoyness) if is_numeric(savoyness) else None
                cupping = int(cupping) if is_numeric(cupping) else None

                return savoyness, cupping

    return None, None

def load_bins(filepath="cupping_bins.npy"):
    try:
        data = np.load(filepath)

        cupping_bins = data["cupping_bins"]
        curvature_bins = data["curvature_bins"]

        return cupping_bins, curvature_bins
    except:
        return None, None

