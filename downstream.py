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


# for this a 3D representation of the leaves is required
def leaf_cupping_multi(leaf_masks, ):
    pass


def leaf_cupping_mono(leaf_masks, mono_depth, n=5, image=None, display=False):

    n = leaf_count_cap(leaf_masks, n)

    for label in range(1, n+1):

        mask = (leaf_masks == label)

        # erode the mask slightly because of noise in the depth mask
        clean_mask = erode_mask(mask, kernel_size=5, iterations=2)

        if display:
            plot_leaf_depth_3d(clean_mask, mono_depth, image=image, disp_mask=mask)

        # need to mask the monocular depth to the current leaf
        # print(f"Mask Shape: {mask.shape}")
        # print(f"Mono Shape: {mono_depth.shape}")
        # print()
        #
        # masked_mono = np.zeros_like(mono_depth)
        # masked_mono[mask] = mono_depth[mask]



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
