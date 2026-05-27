import numpy as np
import os
import cv2
from plots import *
from downstream import savoyness, savoyness_depth, leaf_cupping_mono, leaf_area, savoyness_fft
from constants import *
from util import *
from sklearn.metrics import mean_absolute_error

from dataclasses import dataclass, field
from collections import defaultdict
import random
import sys
import pickle

from detect import score_leaves, order_mask

DISPLAY = False
NUM_LEAVES = 10

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

def cross_validate_scoring(
    plots,
    ground_truth,
    preds,
    n_splits=10,
    test_ratio=0.3,
    keep_num=None
):

    strategies = [
        "raw_mean",
        "raw_median",
        "binned_mean",
        "binned_median"
    ]

    results = {}

    #
    # Initialise storage
    #

    for strategy in strategies:

        results[strategy] = {
            "mae": [],
            "accuracy": [],
            "off_by_one": [],
            "preds": [],
            "labels": []
        }

    #
    # Cross validation
    #

    for seed in range(n_splits):

        # print(f"Split {seed + 1}/{n_splits}")

        train_plots, train_labels, test_plots, test_labels = split_by_plot(
            plots,
            ground_truth,
            test_ratio=test_ratio,
            seed=seed
        )

        train_eval = create_evaluation(
            train_plots,
            train_labels,
            preds,
            keep_num=keep_num
        )

        test_eval = create_evaluation(
            test_plots,
            test_labels,
            preds,
            keep_num=keep_num
        )

        #
        # Fit bins
        #

        bins_med = fit_bins(
            train_eval.medians,
            train_eval.plot_labels
        )

        bins_mean = fit_bins(
            train_eval.means,
            train_eval.plot_labels
        )

        bins_all = fit_bins(
            train_eval.scores,
            train_eval.leaf_labels
        )

        strategy_bins = {
            "raw_mean": bins_mean,
            "raw_median": bins_med,
            "binned_mean": bins_all,
            "binned_median": bins_all
        }

        #
        # Evaluate each strategy
        #

        for strategy in strategies:

            preds_split = predict_image_scores(
                test_eval.scores_list,
                strategy_bins[strategy],
                method=strategy
            )

            metrics = evaluate_predictions(
                preds_split,
                test_eval.plot_labels
            )

            results[strategy]["mae"].append(
                metrics["mae"]
            )

            results[strategy]["accuracy"].append(
                metrics["accuracy"]
            )

            results[strategy]["off_by_one"].append(
                metrics["off_by_one"]
            )

            results[strategy]["preds"].extend(
                preds_split
            )

            results[strategy]["labels"].extend(
                test_eval.plot_labels
            )

    #
    # Convert to arrays
    #

    for strategy in strategies:

        for key in results[strategy]:

            results[strategy][key] = np.asarray(
                results[strategy][key]
            )

    return results



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

    class_means = np.array(class_means, dtype=np.float32)

    valid = np.isfinite(class_means)

    valid_idx = np.where(valid)[0]

    #
    # ONLY fill missing classes
    #

    missing_idx = np.where(~valid)[0]

    interpolated = np.interp(
        missing_idx,
        valid_idx,
        class_means[valid]
    )

    class_means[missing_idx] = interpolated

    #
    # optional monotonic enforcement
    #

    class_means = np.maximum.accumulate(class_means)

    #
    # bins
    #

    bins = []

    for i in range(len(class_means) - 1):

        midpoint = (
            class_means[i]
            + class_means[i + 1]
        ) / 2

        bins.append(midpoint)

    return np.array(bins)

def compute_scores(plots, image_dir, n_leaves=5, attribute="SAVOYNESS", display=False, gt=None):

    res_scores = {}

    n_plots = len(plots)
    plot_count = 0

    for i in range(len(plots)):
        plot = plots[i]
        if gt is not None:
            ground_truth = gt[i]
        else:
            ground_truth = None

        if display and ground_truth:
            print(f"Ground Truth for plot: {ground_truth}")
            print("\n\n")

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
                    elif SAVOYNESS_EVAL_METHOD == "FFT":
                        # use fft method
                        scores = savoyness_fft(detections, image, n=n_leaves, display=display)
                    else:
                        raise ValueError(f"Unsupported Savoyness Method: {SAVOYNESS_EVAL_METHOD}") 

                elif attribute == "CUPPING":
                    # get the cupping scores for the image
                    scores = leaf_cupping_mono(detections, depth, eval=CUPPING_EVAL_METHOD, n=n_leaves,
                                                   image = image, display=display)
                else:
                    raise ValueError(f"Unsupported Leaf Trait: {attribute}")

                if scores is None:
                    continue

                # plot_scores.extend(scores)
                plot_scores.append(scores)

        # res_scores.append(plot_scores)
        res_scores[plot] = plot_scores

    return res_scores


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


def create_evaluation(plots, plot_labels, scores_dict, keep_num=None):
    eval = PlotEvaluation()

    for i in range(len(plots)):
        plot = plots[i]
        label = plot_labels[i]

        scores = scores_dict[plot] # the scores for the given plot in form [[im1_leaf1, im1_leaf2], [im2_leaf1 ..] ...]

        # aggregate all scores from each image together into combined plot scores
        if keep_num is None: 
            combined_scores = np.concatenate(scores)
        else:
            combined_scores = np.concatenate([
                im_scores[:keep_num] for im_scores in scores
            ])

        eval.scores.extend(combined_scores)
        eval.means.append(np.mean(combined_scores))
        eval.medians.append(np.median(combined_scores))
        eval.scores_list.append(combined_scores)

        eval.leaf_labels.extend([label] * len(combined_scores))
        eval.plot_labels.append(label)

    return eval

def load_scores(plots, image_dir=IMAGE_DIR, n_leaves=5, scores_method="SAVOYNESS", display=False, gt=None):

    # compute the scores on the provided plots
    # load the savoyness scores from file if specified:
    if scores_method == "SAVOYNESS":
        file = SAVED_SAVOYNESS_SCORES
    elif scores_method == "CUPPING":
        file = SAVED_CUPPING_SCORES
    else:
        raise ValueError(f"scoring method: {scores_method} not supported")

    if file is not None:
        try:
            with open(file, "rb") as f:
                scores = pickle.load(f)
                print(f"Loaded scores from: {file}")
        except:
            scores = compute_scores(plots, image_dir, n_leaves=n_leaves, attribute=scores_method, display=display, gt=gt)
            with open(file, "wb") as f:
                pickle.dump(scores, f)

    else:
        scores = compute_scores(plots, image_dir, n_leaves=n_leaves, attribute=scores_method, display=display, gt=gt)

    return scores

def evaluate_scoring(plots, ground_truth, preds, test_ratio=0.7, keep_num=None):


    train_plots, train_labels, test_plots, test_labels = split_by_plot(plots, ground_truth, test_ratio=test_ratio)

    train_eval = create_evaluation(train_plots, train_labels, preds, keep_num=keep_num)
    test_eval = create_evaluation(test_plots, test_labels, preds, keep_num=keep_num)

    results_analysis_plot(train_eval, test_eval)

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

    # plot_gt_coverage(ground_truth=savoyness_gt,
    #                  num_classes=9,
    #                  score_type="Savoyness")
    #
    # plot_gt_coverage(ground_truth=cupping_gt,
    #                  num_classes=9,
    #                  score_type="Cupping")

    savoyness_scores = load_scores(savoyness_plots, image_dir=image_dir, n_leaves=n_leaves,
                                   scores_method="SAVOYNESS", display=display, gt=savoyness_gt)

    cupping_scores = load_scores(cupping_plots, image_dir=image_dir, n_leaves=n_leaves,
                                   scores_method="CUPPING", display=display, gt=cupping_gt)

    # now we want to split into train/test given the scores.
    # this is to be done evenly given the distribution of scores

    # run through each of the number of leaves to keep from each image
    leaves_to_keep = [1, 2, 3, 5, 10]

    for keep_num in leaves_to_keep:
        print(f"Evaluating Scoring for {keep_num} leaves per image:")

        # evaluate_scoring(savoyness_plots, savoyness_gt, savoyness_scores, keep_num=keep_num)
        # evaluate_scoring(cupping_plots, cupping_gt, cupping_scores, keep_num=keep_num)

        results = cross_validate_scoring(savoyness_plots, savoyness_gt, savoyness_scores, n_splits=5, test_ratio=0.7, keep_num=keep_num)
        plot_cv_metrics(results, n_leaves=keep_num)

        plot_best_strategy_results(
            results,
            strategy="binned_mean",
            title="Savoyness"
        )


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


def validate_detection(image_dir, num_leaves, annotation_dir=ANNOTATION_DIR, 
                       detection_output=DETECTION_OUTPUT, rescore=False, score_type="CUM", 
                       depth_type="DEPTH_PRO", inset=4, border_distance=4):
    
    num_leaves_requested = 0
    num_leaves_cum = 0
    correct_preds = 0
    iou_scores_cum = []

    image_names = os.listdir(annotation_dir)

    for name in image_names:
        # check to see where the image is
        image_folder = find_image(name, image_dir)

        if image_folder is None:
            continue

        image = load_image(name, image_folder)
        detection = load_image(name, detection_output)

        # re-score the detection if required
        if rescore:
            if depth_type == "DEPTH_PRO":
                depth_map = load_std_depth(name, DEPTH_PRO_DIR)
            else:
                depth_map = load_std_depth(name, MARIGOLD_DIR)

            seg_scores = score_leaves(depth_map=depth_map, segmented_mask=detection,
                                      score_type=score_type, inset=inset, border_distance=border_distance)

            detection = order_mask(detection, seg_scores)

        ground_truth = load_image(name, annotation_dir)
        
        score, leaf_number, iou_scores = validate_predictions(ground_truth, detection, image=image, show=DISPLAY, n=num_leaves, min_score=-1)

        correct_preds += score
        num_leaves_cum += leaf_number
        num_leaves_requested += num_leaves
        iou_scores_cum.extend(iou_scores)

    correct = (correct_preds / num_leaves_cum * 100)
    correct_requested = (correct_preds / num_leaves_requested) * 100
    iou_mean = np.mean(iou_scores_cum)

    # print(f"Correctly detected {correct * 100:.2f}% of leaves\nMean IOU segmentation accuracy: {iou_mean}")
    # print(f"From requested {correct_requested * 100:.2f}")

    return {
        "correct": correct,
        "correct_requested": correct_requested,
        "iou_mean": iou_mean
    }


def main():
    validate_downstream_scoring(IMAGE_DIR, NUM_LEAVES, DATABASE, display=DISPLAY)

    # validation for proposed technique
    # results = validate_detection("../data/left", num_leaves=5,
    #                    annotation_dir=ANNOTATION_DIR,
    #                    detection_output=DETECTION_OUTPUT,
    #                    rescore=True,
    #                    score_type="CUM",
    #                    depth_type="DEPTH_PRO")

    # print("PROPOSED")
    # results = validate_detection(IMAGE_DIR, num_leaves=5,
    #                              annotation_dir=ANNOTATION_DIR,
    #                              detection_output=DETECTION_OUTPUT,
    #                              rescore=True,
    #                              inset=0,
    #                              border_distance=0,
    #                              score_type="CUM",
    #                              depth_type="DEPTH_PRO")
    #
    # print(results)
    #
    #
    # print("SAMV3")
    # results = validate_detection(IMAGE_DIR, num_leaves=5,
    #                    annotation_dir=ANNOTATION_DIR,
    #                    detection_output="./data/samv3_out/merged",
    #                    rescore=True,
    #                    score_type="CUM",
    #                    depth_type="DEPTH_PRO")
    #
    # print(results)

    #
    # print("RCNN")
    # results = validate_detection(IMAGE_DIR, num_leaves=5,
    #                    annotation_dir=ANNOTATION_DIR,
    #                    detection_output="./data/rcnn_detections",
    #                    rescore=True,
    #                    score_type="CUM",
    #                    depth_type="DEPTH_PRO")
    # print(results)
    #
    # print("YOLO")
    # results = validate_detection(IMAGE_DIR, num_leaves=5,
    #                    annotation_dir=ANNOTATION_DIR,
    #                    detection_output="./data/yolo_detections",
    #                    rescore=True,
    #                    score_type="CUM",
    #                    depth_type="DEPTH_PRO")
    # print(results)



if __name__ == "__main__":
    main()
