import numpy as np
import cv2
from plots import *

# naive approach doing this in 2D (should implement this in 3D)
def leaf_area(leaf_masks, n=5):

    n = leaf_count_cap(leaf_masks, n)

    overall_area = 0

    for label in range(1, n+1):
        mask = (leaf_masks == label)

        # get the count of pixels
        pixel_count = np.sum(mask)

        overall_area += pixel_count

    return overall_area / n

def leaf_area_mono(leaf_masks, mono_depth, n=5):
    pass


# for this a 3D representation of the leaves is required
def leaf_cupping_multi(leaf_masks, ):
    pass


def leaf_cupping_mono(leaf_masks, mono_depth, n=5, remove_outliers=False, image=None, display=False):

    n = leaf_count_cap(leaf_masks, n)

    cupping_cum = 0

    for label in range(1, n+1):

        mask = (leaf_masks == label)

        # erode the mask slightly because of noise in the depth mask
        clean_mask = erode_mask(mask, kernel_size=5, iterations=3)

        # extract the points from the monocular depth map
        ys, xs = np.where(clean_mask)

        if len(xs) < 10:
            return 0.0

        zs = mono_depth[ys, xs]

        # remove the outliers
        if remove_outliers:
            med = np.median(zs)
            std = np.std(zs) + 1e-6
            keep = np.abs(zs - med) < 2 * std

            xs = xs[keep]
            ys = ys[keep]
            zs = zs[keep]

        if len(zs) < 10:
            return 0.0
        
        # fit a plane
        A = np.c_[xs, ys, np.ones_like(xs)]
        coeffs, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
        a, b, c = coeffs

        z_plane = a * xs + b * ys + c

        # get the residuals from the plane:
        residuals = zs - z_plane

        # calculate the score
        cupping_score = np.std(residuals)

        # calculate the curvature score:
        A_quad = np.c_[xs**2, ys**2, xs*ys, xs, ys, np.ones_like(xs)]
        coeffs_quad, _, _, _ = np.linalg.lstsq(A_quad, zs, rcond=None)

        qa, qb, qc, qd, qe, qf = coeffs_quad
        curvature_score = np.sqrt(qa**2 + qb**2 + qc**2)

        # normalise the cupping score:
        # depth_range = zs.max() - zs.min() + 1e-6
        # norm_cupping = cupping_score / depth_range

        cupping_cum += cupping_score

        if display:
            print(f"Leaf cupping score: {cupping_score}")
            print(f"Leaf curvature score: {curvature_score}\n")

            # plot_leaf_depth_3d(clean_mask, mono_depth, image=image, disp_mask=mask)
            plot_leaf_from_points(xs, ys, zs, a, b, c, image=image, mask=mask)

    cupping_av = cupping_cum / n

    return cupping_av
        





def erode_mask(mask, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded.astype(bool)


def leaf_count_cap(leaf_masks, n):

    # find if there aren't n unique leaves in the mask
    num_leaves = len(np.unique(leaf_masks)) - 1

    if n >= num_leaves:
        # not enough leaves, reduce the count
        n = num_leaves

    return n
