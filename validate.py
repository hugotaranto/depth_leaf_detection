import numpy as np
import os
import cv2
from plots import *

IMAGE_DIR = "../data/left"
GROUND_TRUTH_DIR = "./annotation_out"
PREDICTED_LEAVES = "./detection_out"
# PREDICTED_LEAVES = "./samv3_out/merged"

DATA_DIR = "../data/left"

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

    return score, iou_result / score

def iou_score(gt_segment, predicted_segment):

    # ensure boolean
    gt_segment = gt_segment.astype(bool)
    predicted_segment = predicted_segment.astype(bool)

    intersection = np.logical_and(gt_segment, predicted_segment).sum()
    union = np.logical_or(gt_segment, predicted_segment).sum()

    if union == 0:
        return 0.0  # avoid division by zero

    return intersection / union


def main():

    # load in the ground truth
    # load in the generated leaves

    # take the top n of the generated leaves

    # see if their leaf is within the ground truth
    # this makes the score

    show = False

    n = 5
    score_cum = 0
    iou_cum = 0
    
    # get the names
    image_names = os.listdir(IMAGE_DIR)
    for name in image_names:
        gt, pred, image = load_gt_pred_pairs(name, GROUND_TRUTH_DIR, PREDICTED_LEAVES, DATA_DIR)

        score, iou_result = validate(gt, pred, image=image, show=show, n=n)
        iou_cum += iou_result
        score_cum += score

        print(f"{score}/{n} leaves detected, IOU average: {iou_result:.4f} : {name}")

    n_images = len(image_names)
    overall_accuracy = (score_cum / n_images) / n
    overall_iou = (iou_cum / n_images)

    print("OVERALL ACCURACY:", overall_accuracy)
    print(f"OVERALL IOU SCORE: {overall_iou:.4f}")


if __name__ == "__main__":
    main()
