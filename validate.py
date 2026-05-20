import numpy as np
import os
import cv2
from plots import *
from downstream import savoyness, savoyness_depth, leaf_cupping_mono, leaf_area
from constants import *
from util import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from dataclasses import dataclass, field
from collections import defaultdict
import random
import sys

SAVOYNESS_TYPE = "RGB"
DISPLAY = False
NUM_LEAVES = 5

@dataclass
class Evaluation:
    savoy_scores: list = field(default_factory=list)
    savoy_means: list = field(default_factory=list)
    savoy_medians: list = field(default_factory=list)
    savoy_scores_list: list = field(default_factory=list)

    savoy_labels: list = field(default_factory=list)
    savoy_image_labels: list = field(default_factory=list)

    cup_scores: list = field(default_factory=list)
    cup_means: list = field(default_factory=list)
    cup_medians: list = field(default_factory=list)
    cup_scores_list: list = field(default_factory=list)

    cup_labels: list = field(default_factory=list)
    cup_image_labels: list = field(default_factory=list)

@dataclass
class PlotEvaluation:
    scores: list = field(default_factory=list)
    means: list = field(default_factory=list)
    medians: list = field(default_factory=list)
    scores_list: list = field(default_factory=list)

    leaf_labels: list = field(default_factory=list)
    plot_labels: list = field(default_factory=list)


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


def validate_predictions(gt, pred, n=5, overlap_thresh=0.5, show=False, image=None, min_score=0.4):

    # ensure single channel
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    if pred.ndim == 3:
        pred = pred[:, :, 0]

    cut_preds = pred.copy()
    cut_preds[cut_preds > n] = 0

    num_leaves = len(np.unique(pred)) - 1 
    n = min(n, num_leaves)
    if n == 0:
        return 0, 0, []

    score = 0
    iou_results = []

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
            # iou_result += iou_score(gt_mask, pred_mask)
            iou_results.append(iou_score(gt_mask, pred_mask))

    # show the masks
    if (show or (score / n) <= min_score) and image is not None:
        display_pred_vs_gt(image, cut_preds, gt)

    return score, n, iou_results

def iou_score(gt_segment, predicted_segment):

    # ensure boolean
    gt_segment = gt_segment.astype(bool)
    predicted_segment = predicted_segment.astype(bool)

    intersection = np.logical_and(gt_segment, predicted_segment).sum()
    union = np.logical_or(gt_segment, predicted_segment).sum()

    if union == 0:
        return 0.0  # avoid division by zero

    return intersection / union


def predict_from_raw(raw_scores, bins, aggregation="median"):

    raw_scores = np.asarray(raw_scores)

    if aggregation == "mean":
        agg = np.mean(raw_scores)

    elif aggregation == "median":
        agg = np.median(raw_scores)

    else:
        raise ValueError("aggregation must be mean or median")

    pred = np.digitize(agg, bins) + 1

    return pred, agg


def predict_from_binned(raw_scores, bins, aggregation="median"):

    leaf_preds = np.digitize(raw_scores, bins) + 1

    if aggregation == "mean":
        pred = int(np.round(np.mean(leaf_preds)))

    elif aggregation == "median":
        pred = int(np.round(np.median(leaf_preds)))

    else:
        raise ValueError("aggregation must be mean or median")

    return pred, leaf_preds


def evaluate_predictions(preds, labels):

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    mae = mean_absolute_error(labels, preds)

    accuracy = np.mean(preds == labels)

    off_by_one = np.mean(
        np.abs(preds - labels) <= 1
    )

    return {
        "mae": mae,
        "accuracy": accuracy,
        "off_by_one": off_by_one
    }


def evaluate_strategies(
    train_scores,
    train_labels,
    test_leaf_scores,
    test_labels
):

    #
    # fit thresholds
    #

    bins = fit_bins(
        train_scores,
        train_labels
    )

    results = {}

    strategies = [
        ("raw_mean", predict_from_raw, "mean"),
        ("raw_median", predict_from_raw, "median"),
        ("binned_mean", predict_from_binned, "mean"),
        ("binned_median", predict_from_binned, "median"),
    ]

    for name, fn, agg in strategies:

        preds = []

        for leaf_scores in test_leaf_scores:

            if leaf_scores is None or len(leaf_scores) == 0:
                continue

            pred, _ = fn(
                leaf_scores,
                bins,
                aggregation=agg
            )

            preds.append(pred)

        metrics = evaluate_predictions(
            preds,
            test_labels[:len(preds)]
        )

        results[name] = metrics

    return bins, results


def predict_image_scores(
    test_leaf_scores,
    bins,
    method="binned_mean"
):

    preds = []

    for scores in test_leaf_scores:

        scores = np.asarray(scores)

        if method == "raw_mean":

            agg = np.mean(scores)
            pred = np.digitize(agg, bins) + 1

        elif method == "raw_median":

            agg = np.median(scores)
            pred = np.digitize(agg, bins) + 1

        elif method == "binned_mean":

            leaf_preds = np.digitize(scores, bins) + 1
            pred = int(np.round(np.mean(leaf_preds)))

        elif method == "binned_median":

            leaf_preds = np.digitize(scores, bins) + 1
            pred = int(np.round(np.median(leaf_preds)))

        else:
            raise ValueError("Unknown method")

        preds.append(pred)

    return np.asarray(preds)


def fit_bins(scores, labels, n_classes=9):

    class_means = []

    for i in range(1, n_classes + 1):

        vals = [
            s for s, l in zip(scores, labels)
            if l == i
        ]

        if len(vals) == 0:
            class_means.append(np.nan)
        else:
            class_means.append(np.median(vals))

    class_means = np.array(class_means)

    #
    # interpolate missing classes
    #

    valid = np.isfinite(class_means)

    class_means = np.interp(
        np.arange(len(class_means)),
        np.where(valid)[0],
        class_means[valid]
    )

    class_means = np.maximum.accumulate(class_means)

    #
    # thresholds between classes
    #

    bins = []

    for i in range(len(class_means) - 1):

        midpoint = (
            class_means[i]
            + class_means[i + 1]
        ) / 2

        bins.append(midpoint)

    return np.array(bins)

def compute_scores(plots, image_dir, n_leaves=5, attribute="SAVOYNESS", display=False):

    res_scores = {}

    n_plots = len(plots)
    plot_count = 0

    for plot in plots:
        if plot_count > 0:
            sys.stdout.write("\x1b[2F")

        plot_count += 1
        sys.stdout.write(f"\nComputing {attribute} scores for plot: {plot} {plot_count}/{n_plots}\n")

        # get the cams
        plot_dir = os.path.join(image_dir, plot)
        cams = os.listdir(plot_dir)

        plot_scores = []

        for cam in cams:
            cam_dir = os.path.join(plot_dir, cam)
            images = os.listdir(cam_dir)

            for im_name in images:

                sys.stdout.write(f"Working on image: {im_name}\r")

                # load and check for 0 leaf detections
                detections = load_image(im_name, DETECTION_OUTPUT)
                if len(np.unique(detections)) < 2:
                    continue

                image = load_image(im_name, cam_dir)

                if DOWNSTREAM_DEPTH_TYPE == "MARIGOLD":
                    depth = load_std_depth(im_name, MARIGOLD_DIR)
                else:
                    depth = load_std_depth(im_name, DEPTH_PRO_DIR)

                if attribute == "SAVOYNESS":
                    # get the savoyness scores for the image
                    if SAVOYNESS_EVAL_METHOD == "LAPLACE":
                        # use laplace method
                        scores = savoyness(detections, image, n=n_leaves, display=display)
                    elif SAVOYNESS_EVAL_METHOD == "DEPTH":
                        # use depth method
                        scores = savoyness_depth(detections, depth, n=n_leaves, display=display, image=image)
                    else:
                        raise ValueError(f"Unsupported Savoyness Method: {SAVOYNESS_EVAL_METHOD}") 

                elif attribute == "CUPPING":
                    # get the cupping scores for the image
                    scores = leaf_cupping_mono(detections, depth, eval=CUPPING_EVAL_METHOD, n=n_leaves,
                                                   image = image, display=display)
                else:
                    raise ValueError(f"Unsupported Attribute: {attribute}")

                if scores is None:
                    continue

                plot_scores.extend(scores)

        # res_scores.append(plot_scores)
        res_scores[plot] = plot_scores

    return res_scores

def eval_images(names, n_leaves, savoyness_type="RGB", display=False):
    eval = Evaluation()

    for name in names:

        image = load_image(name, IMAGE_DIR)
        detections = load_image(name, DETECTION_OUTPUT)

        if DOWNSTREAM_DEPTH_TYPE == "MARIGOLD":
            depth_dir = MARIGOLD_DIR
        else:
            depth_dir = DEPTH_PRO_DIR

        depth = load_depth(name, depth_dir, DOWNSTREAM_DEPTH_TYPE)
        savoyness_gt, cupping_gt = load_eval_scores(name, DATABASE)

        # print(f"Savoyness ground truth: {savoyness_gt}")

        cupping_res = leaf_cupping_mono(detections, depth, curvature_bins=None, cupping_bins=None, n=n_leaves, image=image, display=display)
        if cupping_res is not None:
            _, cupping_scores, _ = cupping_res
        else:
            cupping_scores = None

        if savoyness_type == "RGB":
            savoyness_res = savoyness(detections, image, n=n_leaves, display=display)
        else:
            savoyness_res = savoyness_depth(detections, depth, image=image, display=display)
        if savoyness_res is not None:
            _, savoyness_scores = savoyness_res
        else:
            savoyness_scores = None

        if savoyness_gt is not None and savoyness_scores is not None:
            eval.savoy_scores_list.append(savoyness_scores)

            eval.savoy_scores.extend(savoyness_scores)
            eval.savoy_labels.extend([savoyness_gt] * len(savoyness_scores))

            mean_savoyness = np.mean(savoyness_scores)
            median_savoyness = np.median(savoyness_scores)

            eval.savoy_means.append(mean_savoyness)
            eval.savoy_medians.append(median_savoyness)

            eval.savoy_image_labels.append(savoyness_gt)

        if cupping_gt is not None and cupping_scores is not None:
            eval.cup_scores_list.append(cupping_scores)

            eval.cup_scores.extend(cupping_scores)
            eval.cup_labels.extend([cupping_gt] * len(cupping_scores))

            mean_cupping = np.mean(cupping_scores)
            median_cupping = np.median(cupping_scores)

            eval.cup_means.append(mean_cupping)
            eval.cup_medians.append(median_cupping)

            eval.cup_image_labels.append(cupping_gt)

    return eval

def results_analysis(train_eval, test_eval):
    # FIT using medians
    bins_med, results_med = evaluate_strategies(
        train_scores=train_eval.savoy_medians,
        train_labels=train_eval.savoy_image_labels,
        test_leaf_scores=test_eval.savoy_scores_list,
        test_labels=test_eval.savoy_image_labels
    )
    print("Fit with medians")
    print(results_med, "\n\n")

    #FIT using means
    bins_mean, results_mean = evaluate_strategies(
        train_scores=train_eval.savoy_means,
        train_labels=train_eval.savoy_image_labels,
        test_leaf_scores=test_eval.savoy_scores_list,
        test_labels=test_eval.savoy_image_labels
    )
    print("Fit with means:")
    print(results_mean, "\n\n")

    # Fit using all leaves
    bins_all, results_all = evaluate_strategies(
        train_scores=train_eval.savoy_scores,
        train_labels=train_eval.savoy_labels,
        test_leaf_scores=test_eval.savoy_scores_list,
        test_labels=test_eval.savoy_image_labels
    )
    print("Fit with all leaves:")
    print(results_all)

    plot_strategy_comparison(
        results_med,
        results_mean,
        results_all
    )

    plot_bins(
        train_eval.savoy_scores,
        train_eval.savoy_labels,
        bins_all,
        title="Savoyness Thresholds"
    )

    preds = predict_image_scores(
        test_eval.savoy_scores_list,
        bins_all,
        method="binned_mean"
    )

    plot_prediction_scatter(
        test_eval.savoy_image_labels,
        preds,
        title="Savoyness Predictions"
    )

    plot_confusion(
        test_eval.savoy_image_labels,
        preds,
        n_classes=9,
        title="Savoyness Confusion Matrix"
    )

    plot_leaf_score_distributions(
        test_eval.savoy_scores_list,
        test_eval.savoy_image_labels
    )

def split_by_plot(plots, labels, test_ratio=0.2, seed=10):

    random.seed(seed)
    
    score_dict = defaultdict(list)

    for i in range(len(plots)):
        plot = plots[i]
        label = labels[i]

        if label == None:
            continue

        score_dict[label].append(plot)

    train_plots = []
    train_labels = []

    test_plots = []
    test_labels = []

    for score, plots in score_dict.items():
        plots = list(plots)
        random.shuffle(plots)

        n_test = max(1, int(len(plots) * test_ratio))

        test_split = plots[:n_test]
        train_split = plots[n_test:]

        train_plots.extend(train_split)
        test_plots.extend(test_split)

        train_labels.extend([score] * len(train_split))
        test_labels.extend([score] * len(test_split))

    return train_plots, train_labels, test_plots, test_labels


def create_evaluation(plots, plot_labels, scores_dict):
    eval = PlotEvaluation()

    for i in range(len(plots)):
        plot = plots[i]
        label = plot_labels[i]

        scores = scores_dict[plot] # the scores for the given plot
        eval.scores.extend(scores)
        eval.means.append(np.mean(scores))
        eval.medians.append(np.median(scores))
        eval.scores_list.append(scores)

        eval.leaf_labels.extend([label] * len(scores))
        eval.plot_labels.append(label)

    return eval


def validate_downstream_scoring(image_dir, n_leaves, database_file, display=False):

    plots = np.sort(os.listdir(image_dir))
    savoyness_gt = []
    cupping_gt = []

    savoyness_plots = []
    cupping_plots = []

    for plot in plots:
        # get the plot number
        plot_number = plot[-3:]
        try:
            plot_number = int(plot_number)
        except:
            continue

        # now load the eval scores for this plot
        savoyness, cupping = load_plot_eval_scores(plot_number, database_file)
        # print(f"Plot {plot_number} savoyness: {savoyness_gt}, cupping: {cupping_gt}")

        if savoyness is not None:
            savoyness_plots.append(plot)
            savoyness_gt.append(savoyness)

        if cupping is not None: 
            cupping_plots.append(plot)
            cupping_gt.append(cupping)

    # compute the scores on these plots
    savoyness_scores = compute_scores(savoyness_plots, image_dir, n_leaves=n_leaves, attribute="SAVOYNESS", display=display)



    # now we want to split into train/test given the scores.
    # this is to be done evenly given the distribution of scores
    train_plots, train_labels, test_plots, test_labels = split_by_plot(savoyness_plots, savoyness_gt, test_ratio=0.2)

    train_eval = create_evaluation(train_plots, train_labels, savoyness_scores)
    test_eval = create_evaluation(test_plots, test_labels, savoyness_scores)

    results_analysis_plot(train_eval, test_eval)


def results_analysis_plot(train_eval:PlotEvaluation, test_eval:PlotEvaluation):

    bins_med, results_med = evaluate_strategies(
        train_scores=train_eval.medians,
        train_labels=train_eval.plot_labels,
        test_leaf_scores=test_eval.scores_list,
        test_labels=test_eval.plot_labels
    )

    print("Fit with medians")
    print(results_med, "\n\n")

    #FIT using means
    bins_mean, results_mean = evaluate_strategies(
        train_scores=train_eval.means,
        train_labels=train_eval.plot_labels,
        test_leaf_scores=test_eval.scores_list,
        test_labels=test_eval.plot_labels
    )
    print("Fit with means:")
    print(results_mean, "\n\n")

    # Fit using all leaves
    bins_all, results_all = evaluate_strategies(
        train_scores=train_eval.scores,
        train_labels=train_eval.leaf_labels,
        test_leaf_scores=test_eval.scores_list,
        test_labels=test_eval.plot_labels
    )
    print("Fit with all leaves:")
    print(results_all)

    plot_strategy_comparison(
        results_med,
        results_mean,
        results_all
    )

    plot_bins(
        train_eval.scores,
        train_eval.leaf_labels,
        bins_all,
        title="Savoyness Thresholds"
    )

    preds = predict_image_scores(
        test_eval.scores_list,
        bins_all,
        method="binned_mean"
    )

    plot_prediction_scatter(
        test_eval.plot_labels,
        preds,
        title="Savoyness Predictions"
    )

    plot_confusion(
        test_eval.plot_labels,
        preds,
        n_classes=9,
        title="Savoyness Confusion Matrix"
    )

    plot_leaf_score_distributions(
        test_eval.scores_list,
        test_eval.plot_labels
    )



# def validate_downstream_scoring(image_dir, n_leaves, database_file):
#
#     # get the image names
#     image_names = os.listdir(image_dir)
#
#     valid_images = []
#     for name in image_names:
#         savoyness_gt, cupping = load_eval_scores(name, database_file)
#
#         if savoyness_gt is None and cupping is None:
#             continue
#         
#         valid_images.append(name)
#
#     # split the data into train/test
#     train_names, test_names = train_test_split(valid_images, test_size=0.2, random_state=10)
#
#     # build the train data
#     train_eval = eval_images(train_names, n_leaves, savoyness_type=SAVOYNESS_TYPE, display=DISPLAY)
#     test_eval = eval_images(test_names, n_leaves, savoyness_type=SAVOYNESS_TYPE, display=DISPLAY)
#
#     results_analysis(train_eval, test_eval)

def validate_detection(image_dir, num_leaves):
    
    num_leaves_requested = 0
    num_leaves_cum = 0
    correct_preds = 0
    iou_scores_cum = []

    image_names = os.listdir(ANNOTATION_DIR)

    for name in image_names:
        # check to see where the image is
        image_folder = find_image(name, image_dir)
        if image_folder is None:
            continue

        image = load_image(name, image_folder)
        detection = load_image(name, DETECTION_OUTPUT)
        ground_truth = load_image(name, ANNOTATION_DIR)
        
        score, leaf_number, iou_scores = validate_predictions(ground_truth, detection, image=image, show=DISPLAY, n=num_leaves, min_score=-1)

        correct_preds += score
        num_leaves_cum += leaf_number
        num_leaves_requested += num_leaves
        iou_scores_cum.extend(iou_scores)

    correct = correct_preds / num_leaves_cum
    correct_requested = correct_preds / num_leaves_requested
    iou_mean = np.mean(iou_scores_cum)

    print(f"Correctly detected {correct * 100:.2f}% of leaves\nMean IOU segmentation accuracy: {iou_mean}")
    print(f"From requested {correct_requested * 100:.2f}")


def main():
    # validate_detection(IMAGE_DIR, NUM_LEAVES)
    validate_downstream_scoring(IMAGE_DIR, NUM_LEAVES, DATABASE, display=DISPLAY)

if __name__ == "__main__":
    main()
