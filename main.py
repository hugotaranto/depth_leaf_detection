import os
import numpy as np
from constants import *
from compute_depths import depth_pro_predict_image, load_depth_pro, MarigoldDepthEstimator, marigold_predict_image
from detect import load_sam, detect_plot
from validate import compute_plot_scores, get_plots_gt, cross_validate_scoring
from plots import *
import torch
import warnings
from diffusers.utils import logging

logging.set_verbosity_error()

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

warnings.filterwarnings(
    "ignore",
    message=".*prediction_type.*"
)

def get_image_paths_from_plot(plot_dir):

    cams = os.listdir(plot_dir)

    # load all of the image paths/names from this plot
    image_paths = []

    for cam in cams:
        cam_dir = os.path.join(plot_dir, cam)
        images = os.listdir(cam_dir)

        for image in images:
            image_path = os.path.join(cam_dir, image)
            
            image_paths.append((image_path, image))

    return image_paths


def compute_depth_maps(plot_dir):

    image_paths = get_image_paths_from_plot(plot_dir)

    # compute the depth maps for all images in this plot
    print(f"Computing Depth Pro maps for images in plot: {plot_dir}\n")

    dp_model, dp_transform = load_depth_pro(config=DEPTH_PRO_CONFIG)
    depth_pro_depths = {}

    for im in image_paths:
        path, name = im
        depth = depth_pro_predict_image(path, name, dp_model, 
                                        dp_transform, save_path=None)

        depth_pro_depths[name] = depth

    del dp_model, dp_transform
    torch.cuda.empty_cache()

    print(f"Computing Marigold depth maps for images in plot: {plot_dir}\n")
    marigold_model = MarigoldDepthEstimator(checkpoint=MARIGOLD_CHECKPOINT)
    marigold_depths = {}

    for im in image_paths:
        path, name = im
        depth = marigold_predict_image(path, name, marigold_model, save_path=None)

        marigold_depths[name] = depth

    del marigold_model
    torch.cuda.empty_cache()

    return depth_pro_depths, marigold_depths


def main(image_dir):

    DISPLAY = False

    # number of leaves per image to analyse
    NUM_LEAVES = 5
    plots = sorted(os.listdir(image_dir))

    savoyness_scores = {}
    cupping_scores = {}

    # process each plot
    for plot in plots:
        plot_dir = os.path.join(image_dir, plot)

        # compute the depth maps
        depth_pro_depths, marigold_depths = compute_depth_maps(plot_dir)

        # perform leaf detection on this plot 
        sam_predictor = load_sam(SAM_PATH, SAM_MODEL_TYPE)
        segmentation_masks = detect_plot(plot_dir, sam_predictor, save_dir=DETECTION_OUTPUT,
                                         marigold_depths=marigold_depths, depth_pro_depths=depth_pro_depths)

        # then perform trait analysis
        sav_scores = compute_plot_scores(plot_dir, n_leaves=NUM_LEAVES,
                                         attribute="SAVOYNESS", display=DISPLAY,
                                         marigold_depths=marigold_depths,
                                         depth_pro_depths=depth_pro_depths,
                                         leaf_segs=segmentation_masks)

        cup_scores = compute_plot_scores(plot_dir, n_leaves=NUM_LEAVES,
                                         attribute="CUPPING", display=DISPLAY,
                                         marigold_depths=marigold_depths,
                                         depth_pro_depths=depth_pro_depths,
                                         leaf_segs=segmentation_masks)

        savoyness_scores[plot] = sav_scores
        cupping_scores[plot] = cup_scores
        

    # validate the savoyness scoring
    savoyness_plots, savoyness_gt, cupping_plots, cupping_gt = get_plots_gt(image_dir, DATABASE)

    savoyness_results = cross_validate_scoring(savoyness_plots, savoyness_gt, 
                                               savoyness_scores, n_splits=5,
                                               test_ratio=0.7)

    # compare the fitting strategies
    plot_fit_strategy_metrics(
        savoyness_results,
        n_leaves=NUM_LEAVES,
        title=f"Fitting Strategy Comparison For {SAVOYNESS_EVAL_METHOD} Savoyness"
    )

    # compare prediction strategies within a selected fitting method
    plot_prediction_strategy_metrics(
        savoyness_results,
        fit_strategy="fit_means",
        title=f"Fit Means Prediction Strategy Comparison For {SAVOYNESS_EVAL_METHOD} Savoyness"
    )

    # validate the cupping scoring
    results = cross_validate_scoring(cupping_plots, cupping_gt, cupping_scores, n_splits=5, test_ratio=0.7)

    # compare the fitting strategies
    plot_fit_strategy_metrics(
        results,
        n_leaves=NUM_LEAVES,
        title=f"Fitting Strategy Comparison For {CUPPING_EVAL_METHOD} Cupping"
    )

    # compare prediction strategies within a selected fitting method
    plot_prediction_strategy_metrics(
        results,
        fit_strategy="fit_means",
        title=f"Fit Means Prediction Strategy Comparison For {CUPPING_EVAL_METHOD} Cupping"
    )

if __name__ == "__main__":
    main(IMAGE_DIR)
