import numpy as np
import os
import cv2
from plots import *
from downstream import leaf_area, leaf_cupping_mono, leaf_cupping_multi

IMAGE_DIR = "../data/left"
GROUND_TRUTH_DIR = "./annotation_out"
PREDICTED_LEAVES = "./detection_out/left_new"
# PREDICTED_LEAVES = "./samv3_out/merged"

MONOCULAR_DEPTH_DIR = "./mono_depths/depth_pro"
MONO_DEPTH_TYPE = "DEPTH_PRO"

# MONOCULAR_DEPTH_DIR = "./mono_depths/marigold"
# MONO_DEPTH_TYPE = "MARIGOLD"

DATA_DIR = "../data/left"
BINS_FILE = "./bins.npz"

def load_gt_pred_pairs(name, gt_path, pred_path, image_path):
    # load in each mask
    gt_image = os.path.join(gt_path, name)
    pred_image = os.path.join(pred_path, name)
    image_path = os.path.join(image_path, name)

    gt = cv2.imread(gt_image, cv2.IMREAD_UNCHANGED)
    pred = cv2.imread(pred_image, cv2.IMREAD_UNCHANGED)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if gt is None or pred is None or image is None:
        raise RuntimeError(f"Could not read image: {name}")

    return gt, pred, image

def load_mono_depth(name, data_dir, depth_type):

    name = os.path.splitext(name)[0]
    
    if depth_type == "MARIGOLD":
        # construct the depth name
        depth_name = f"{name}_depth.npy"
    elif depth_type == "DEPTH_PRO":
        depth_name = f"{name}.npz"

    else:
        raise RuntimeError(f"Depth type: {depth_type} no supported")

    depth_path = os.path.join(data_dir, depth_name)


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


def validate(gt, pred, n=5, overlap_thresh=0.5, show=False, image=None):
    """
    gt: ground truth mask (H, W) with labels
    pred: predicted mask (H, W) with labels ranked 1..N
    n: number of top predicted leaves to check
    overlap_thresh: fraction required to count as match
    """

    # ensure single channel
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    # print("Num Predictions", np.max(pred))

    cut_preds = pred.copy()
    cut_preds[cut_preds > n] = 0

    # print(np.unique(cut_preds))

    # show the masks
    if show and image is not None:
        display_pred_vs_gt(image, cut_preds, gt)
        pass

    score = 0
    iou_result = 0

    for label in range(1, n + 1):
        pred_mask = (pred == label)

        if np.sum(pred_mask) == 0:
            continue  # skip empty predictions

        # get GT labels overlapping this prediction
        overlapping_gt = gt[pred_mask]

        # ignore background (assume 0 = background)
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        if len(overlapping_gt) == 0:
            continue

        # find most common GT label
        gt_label, count = np.unique(overlapping_gt, return_counts=True)

        max_overlap = 0
        index = -1
        for i in range(len(count)):
            if count[i] > max_overlap:
                max_overlap = count[i]
                index = i

        max_label = gt_label[index]
        gt_mask = (gt == max_label)

        # fraction of predicted segment that overlaps best GT leaf
        overlap_ratio = max_overlap / np.sum(pred_mask)

        if overlap_ratio >= overlap_thresh:
            score += 1

            # get the iou score
            iou_result += iou_score(gt_mask, pred_mask)

    if score == 0:
        iou_result = 0
    else:
        iou_result = iou_result / score

    return score, iou_result

def iou_score(gt_segment, predicted_segment):

    # ensure boolean
    gt_segment = gt_segment.astype(bool)
    predicted_segment = predicted_segment.astype(bool)

    intersection = np.logical_and(gt_segment, predicted_segment).sum()
    union = np.logical_or(gt_segment, predicted_segment).sum()

    if union == 0:
        return 0.0  # avoid division by zero

    return intersection / union


def compute_bins(scores, n_bins=10):
    scores = np.asarray(scores)

    # percentiles from 0 → 100
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(scores, percentiles)

    # ensure strictly increasing (handles duplicates)
    bins = np.unique(bins)

    return bins

def save_cupping_curvature_bins(cupping_bins, curvature_bins, filepath="leaf_bins.npz"):
    np.savez(
        filepath,
        cupping_bins=np.asarray(cupping_bins),
        curvature_bins=np.asarray(curvature_bins)
    )


def load_bins(filepath="cupping_bins.npy"):
    try:
        data = np.load(filepath)

        cupping_bins = data["cupping_bins"]
        curvature_bins = data["curvature_bins"]

        return cupping_bins, curvature_bins
    except:
        return None, None

def main():

    # load in the ground truth
    # load in the generated leaves

    # take the top n of the generated leaves

    # see if their leaf is within the ground truth
    # this makes the score

    show = True

    n = 5
    score_cum = 0
    iou_cum = 0

    cup_scores = []
    curve_scores = []

    cupping_bins, curvature_bins = load_bins(BINS_FILE)

    # get the names
    image_names = os.listdir(IMAGE_DIR)
    for name in image_names:
        gt, pred, image = load_gt_pred_pairs(name, GROUND_TRUTH_DIR, PREDICTED_LEAVES, DATA_DIR)

        score, iou_result = validate(gt, pred, image=image, show=show, n=n)
        iou_cum += iou_result
        score_cum += score

        print(f"{score}/{n} leaves detected, IOU average: {iou_result:.4f} : {name}")

        # get the average leaf area
        av_area = leaf_area(pred, n=n)

        print(f"Average leaf area: {av_area:.1f} Px")
        print()

        # Calculate the leaf cupping
        mono_depth = load_mono_depth(name, MONOCULAR_DEPTH_DIR, MONO_DEPTH_TYPE)

        cupping_av, cupping_scores, curvature_scores = leaf_cupping_mono(pred, mono_depth, curvature_bins, cupping_bins, n, image=image, display=False)

        cup_scores.extend(cupping_scores)
        curve_scores.extend(curvature_scores)


    n_images = len(image_names)
    overall_accuracy = (score_cum / n_images) / n
    overall_iou = (iou_cum / n_images)

    print("OVERALL ACCURACY:", overall_accuracy)
    print(f"OVERALL IOU SCORE: {overall_iou:.4f}")

    cupping_bins = compute_bins(cup_scores, 10)
    curve_bins = compute_bins(curve_scores, 10)

    save_cupping_curvature_bins(cupping_bins, curve_bins, filepath=BINS_FILE)


if __name__ == "__main__":
    main()
