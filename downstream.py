import numpy as np
import cv2
from plots import *
from util import *

from constants import *

# naive approach doing this in 2D (should implement this in 3D)
def leaf_area(leaf_masks, n=5):

    n = leaf_count_cap(leaf_masks, n)

    if n == 0 or n == None:
        return None

    overall_area = 0

    for label in range(1, n+1):
        mask = (leaf_masks == label)

        # get the count of pixels
        pixel_count = np.sum(mask)

        overall_area += pixel_count

    return overall_area / n

def savoyness_fft(
    leaf_masks,
    image,
    n=5,
    median_kernel=5,
    blur_sigma=8,
    patch_size=50,
    n_patches=10,
    low_cut=4,
    high_cut=16,
    display=False
):
    n = leaf_count_cap(leaf_masks, n)

    if n == 0 or n == None:
        return None

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L = lab[:, :, 0].astype(np.float32)

    #
    # Median filter
    #

    L_med = cv2.medianBlur(
        L.astype(np.uint8),
        median_kernel
    ).astype(np.float32)

    #
    # Remove low-frequency illumination
    #

    smooth = cv2.GaussianBlur(
        L_med,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma
    )

    texture = L_med - smooth

    half = patch_size // 2
    scores = []

    corner_distance = np.sqrt(2 * half**2)

    for label in range(1, n + 1):

        mask = (leaf_masks == label)

        # erode the mask to remove the noisy edges
        clean_mask = erode_mask(
            mask,
            kernel_size=5,
            iterations=2
        )

        #
        # Find the patches from within the leaf
        #

        # use distance transform to find where valid centers are
        distance = cv2.distanceTransform(
            clean_mask.astype(np.uint8),
            cv2.DIST_L2,
            5
        )

        valid_mask = distance >= corner_distance

        ys, xs = np.where(valid_mask)       

        if len(xs) == 0:
            continue

        # randomly select the patch centers
        num_patches = min(n_patches, len(xs))
        idx = np.random.choice(len(xs), size=num_patches, replace=False)

        patch_scores = []

        for i in idx:

            x = xs[i]
            y = ys[i]

            patch = texture[
                y-half:y+half,
                x-half:x+half
            ]

            if patch.shape[:2] != (patch_size, patch_size):
                continue

            # perform the windowing on this patch to smooth the edges
            window_han = np.hanning(patch_size)

            window = np.outer(window_han, window_han)
            patch_windowed = patch * window

            # perform the fft on this patch
            fft = np.fft.fft2(patch_windowed)
            fft_shift = np.fft.fftshift(fft)

            magnitude = np.log1p(np.abs(fft_shift))

            h_fft, w_fft = magnitude.shape

            cy = h_fft // 2
            cx = w_fft // 2

            yy, xx = np.ogrid[:h_fft, :w_fft]

            # Frequency radius
            radius = np.sqrt((yy - cy)**2 + (xx - cx)**2)

            freq_mask = (
                (radius >= low_cut) &
                (radius <= high_cut)
            )

            vals = magnitude[freq_mask]
            total_vals = magnitude[radius >= low_cut]

            if len(vals) == 0 or len(total_vals) == 0:
                continue

            band_energy = np.mean(vals)
            total_energy = np.mean(total_vals)
            score = band_energy / (total_energy + 1e-8)

            patch_scores.append(score)

            if display:

                plot_fft_savoyness_process(
                    image=image,
                    mask=mask,
                    texture=texture,
                    patch_center=(x, y),
                    patch_size=patch_size,
                    magnitude=magnitude,
                    freq_mask=freq_mask
                )

        scores.append(np.median(patch_scores))

        if display:
            print("\n\n")
            print(f"Computed FFT score for leaf: {np.median(patch_scores)}")

    return scores
    

def savoyness(
    leaf_masks,
    image,
    n=5,
    median_kernel=5,
    blur_sigma=8,
    display=False,
):

    n = leaf_count_cap(leaf_masks, n)

    if n == 0:
        return None

    #
    # RGB -> LAB
    #

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L = lab[:, :, 0].astype(np.float32)

    #
    # Median filter
    #

    L_med = cv2.medianBlur(
        L.astype(np.uint8),
        median_kernel
    ).astype(np.float32)

    #
    # Remove low-frequency illumination
    #

    smooth = cv2.GaussianBlur(
        L_med,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma
    )

    texture = L_med - smooth

    #
    # Laplacian texture energy
    #

    lap = cv2.Laplacian(
        texture,
        cv2.CV_32F,
        ksize=3
    )

    scores = []

    for label in range(1, n + 1):

        mask = (leaf_masks == label)

        clean_mask = erode_mask(
            mask,
            kernel_size=5,
            iterations=2
        )

        if np.count_nonzero(clean_mask) < 20:
            continue

        vals = L_med[clean_mask]

        if len(vals) < 20:
            continue

        #
        # Remove brightest pixels
        #

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        S = hsv[:,:,1].astype(np.float32)
        V = hsv[:,:,2].astype(np.float32)

        artifact_mask = (
            (V > 190) & (S < 30)
        )

        valid_mask = clean_mask & (~artifact_mask)

        if np.count_nonzero(valid_mask) < 20:
            continue

        # score the savoyness based on the median of the laplacian
        lap_vals = np.abs(lap[valid_mask])

        score = np.median(lap_vals)

        scores.append(score)

        if display:
            print("\n\n")
            print(f"Leaf {label} savoyness: {score:.4f}")
            # plot_savoyness_process(mask, image, texture, valid_mask, clean_mask, lap)
            plot_savoyness_process_grid(mask, image, texture, valid_mask, clean_mask, lap)
            # plot_selected_leaf(mask, image)


    # return np.mean(scores), scores
    return scores


def savoyness_depth(
    leaf_masks,
    mono_depth,
    sigma=9,
    n=5,
    remove_outliers=True,
    display=False,
    image=None
):

    n = leaf_count_cap(leaf_masks, n)

    scores = []

    for label in range(1, n + 1):

        mask = (leaf_masks == label)

        clean_mask = erode_mask(
            mask,
            kernel_size=5,
            iterations=2
        )

        ys, xs = np.where(clean_mask)

        if len(xs) < 20:
            continue

        depth_vals = mono_depth[ys, xs]

        valid = np.isfinite(depth_vals)

        ys = ys[valid]
        xs = xs[valid]
        depth_vals = depth_vals[valid]

        if len(depth_vals) < 20:
            continue

        masked_depth = mono_depth * mask
        blurred_depth = cv2.GaussianBlur(
            masked_depth.astype(np.float32),
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )
        blurred_mask = cv2.GaussianBlur(
            mask.astype(np.float32),
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )

        smooth_depth = blurred_depth / (blurred_mask + 1e-6)

        zs = mono_depth[ys, xs]
        smooth_zs = smooth_depth[ys, xs]

        #
        # residual texture
        #

        residuals = zs - smooth_zs

        #
        # optional outlier removal
        #

        if remove_outliers:

            med = np.median(residuals)
            mad = np.median(np.abs(residuals - med)) + 1e-6

            keep = (
                np.abs(residuals - med)
                < 3 * mad
            )

            residuals = residuals[keep]
            xs = xs[keep]
            ys = ys[keep]
            zs = zs[keep]
            smooth_zs = smooth_zs[keep]

        #
        # savoyness score
        #

        score = np.std(residuals)

        scores.append(score)

        if display:
            print("\n\n")
            print(f"Savoyness: {score:.4f}")
            plot_leaf_savoyness(xs, ys, zs, smooth_zs, residuals, image=image, mask=mask)

    return scores

def assign_bin(score, bins):
    idx = np.digitize(score, bins, right=True)
    return max(1, min(idx, len(bins) - 1))

def leaf_cupping_mono(leaf_masks, mono_depth, eval="QUADRATIC", n=5, remove_outliers=False, image=None, display=False):

    if eval not in ["PLANE", "QUADRATIC"]:
        raise ValueError(f"Unsupported Cupping Method: {CUPPING_EVAL_METHOD}")

    n = leaf_count_cap(leaf_masks, n)
    if n == 0:
        return None

    scores = []

    for label in range(1, n+1):

        mask = (leaf_masks == label)

        # erode the mask slightly because of noise in the depth mask
        clean_mask = erode_mask(mask, kernel_size=5, iterations=3)

        # extract the points from the monocular depth map
        ys, xs = np.where(clean_mask)

        if len(xs) < 10:
            continue

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
            continue
        
        if eval == "PLANE":
            # fit a plane
            A = np.c_[xs, ys, np.ones_like(xs)]
            coeffs, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
            a, b, c = coeffs

            z_plane = a * xs + b * ys + c

            # get the residuals from the plane:
            residuals = zs - z_plane

            # calculate the score
            score = np.std(residuals)
            scores.append(score)
            if display:
                print("\n\n")
                print(f"Leaf {eval} score: {score}")
                plot_leaf_from_points(xs, ys, zs, a, b, c, image=image, mask=mask)

        else:

            # calculate the curvature score:
            A_quad = np.c_[xs**2, ys**2, xs*ys, xs, ys, np.ones_like(xs)]
            coeffs_quad, _, _, _ = np.linalg.lstsq(A_quad, zs, rcond=None)

            qa, qb, qc, qd, qe, qf = coeffs_quad
            score = np.sqrt(qa**2 + qb**2 + qc**2)

            scores.append(score)
            if display:
                print("\n\n")
                print(f"Leaf {eval} score: {score}")
                plot_leaf_quadratic(xs, ys, zs, coeffs_quad, image, mask=mask)

    return scores

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
