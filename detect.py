import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from segment_anything import sam_model_registry, SamPredictor
import torch

from plots import *

IMAGE_DATA_DIR = '../data/left'

# DEPTH_TYPE = "DEPTH_PRO"
# DEPTH_DATA_DIR = './mono_depths/depth_pro'

DEPTH_TYPE = "MARIGOLD"
DEPTH_DATA_DIR = '../Marigold/output/depth_npy'

DOWNSAMPLE_SIZE = 256

# base SAM model
SAM_MODEL_TYPE = 'vit_l'
# SAM_MODEL_PATH = './sam_base_checkpoint.pth'
SAM_MODEL_PATH = './sam_checkpoints/sam_vit_l_0b3195.pth'

OUTPUT_DIR = "./detection_out"

# load the data into a dictionary
def load_data(image_dir, depth_dir):

    # get all of the image names
    image_names = os.listdir(image_dir)

    image_depth_pairs = []

    for name in image_names:
        base_name = os.path.splitext(name)[0] # take the .png off

        if DEPTH_TYPE == "MARIGOLD":
            # construct the depth name
            depth_name = f"{base_name}_depth.npy"
        elif DEPTH_TYPE == "DEPTH_PRO":
            depth_name = f"{base_name}.npz"

        else:
            raise RuntimeError(f"Depth type: {DEPTH_TYPE} no supported")

        # create the full paths
        image_path = os.path.join(image_dir, name)
        depth_path = os.path.join(depth_dir, depth_name)

        if not os.path.exists(depth_path):
            print(f"Could not find depth corresponding depth file: {depth_path}")
            continue

        # load the image and depth map
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        if DEPTH_TYPE == "MARIGOLD":
            depth = np.load(depth_path).astype(np.float32)
        elif DEPTH_TYPE == "DEPTH_PRO":
            depth = np.load(depth_path)
            depth = depth["depth"].astype(np.float32)
        else:
            raise RuntimeError(f"Depth type: {DEPTH_TYPE} no supported")

        image_depth_pairs.append((image, depth, name))

    return image_depth_pairs
 

# use Kmeans to separate the background from the foreground (needs some work)
def get_foreground_mask(depth_map):
    depth_vals = depth_map.ravel().reshape(-1, 1)

    kmeans = KMeans(n_clusters=2).fit(depth_vals)
    labels = kmeans.labels_

    # the cluster with the smaller mean depth (closer to camera) is the leaves/foreground

    cluster_means = [depth_vals[labels==i].mean() for i in range(2)]
    leaf_cluster = np.argmin(cluster_means)

    mask = labels.reshape(depth_map.shape) == leaf_cluster
    return mask

def get_foreground_mask_colour(image):
    # Convert image (RGB) to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Use only H + S channels for clustering
    hs = hsv[:, :, :2].reshape(-1, 2).astype(np.float32)

    # Run KMeans on (H, S)
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(hs)
    centers = kmeans.cluster_centers_  # shape: (2, 2) -> (H, S)

    green_cluster = np.argmin(np.abs(centers[:, 0] - 60))

    # Build mask
    mask = (labels.reshape(image.shape[:2]) == green_cluster).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)).astype(bool)

    return mask

# get the centroids in 3d space
def get_dbscan_centroids(points, labels):

    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        if label == -1:
            # skip the noise labels
            continue

        cluster_points = points[labels == label]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)

    return np.array(centroids)

# dbscan to find the leaf clusters
def dbscan(depth_map, image, show=False):
    
    # resize the image -- DBSCAN struggles with native 1900 * 1900
    resized = cv2.resize(depth_map, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE))
    resized_image = cv2.resize(image, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE))

    # kmeans to separate plant from soil based on colour
    mask = get_foreground_mask_colour(resized_image)

    if show:
        plt.subplot(1, 2, 1)
        plt.imshow(resized_image)

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.show()

    height, width = resized.shape
    yy, xx = np.mgrid[0:height, 0:width]

    # this is the coordinates of the points in 3d space
    coords = np.column_stack((xx.ravel(), yy.ravel(), resized.ravel()))
    filtered_points = coords[mask.ravel()].astype(np.float32)

    # scale the points
    scaled_points = StandardScaler().fit_transform(filtered_points)

    DEPTH_WEIGHT = 2.0
    scaled_points[:, 2] *= DEPTH_WEIGHT

    # DBSCAN parameters
    # eps = 0.05
    # min_samples = 20

    eps = 0.05
    min_samples = 15

    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(scaled_points)

    labels = db.labels_

    # get the centroids
    centroids = get_dbscan_centroids(filtered_points, labels)

    if len(centroids) == 0:
        return centroids

    orig_centroids = centroids.copy()

    # scale the centroids back to the original image
    original_height, original_width = depth_map.shape

    scale_x = float(original_width) / float(width)
    scale_y = float(original_height) / float(height)

    centroids[:, 0] *= scale_x
    centroids[:, 1] *= scale_y

    if show:
        show_dbscan_clusters(resized, filtered_points[:, :2], labels, image, depth_map, centroids, orig_centroids)

    return centroids

# segment the leaves with sam
def segment_with_sam(image, centroids, predictor):
     
    predictor.set_image(image)

    height, width = image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    leaf_masks = []

    leaf_index = 1

    # segment each leaf using the centroids
    for i, (x, y) in enumerate(centroids[:, :2].astype(int)):
        input_point = np.array([[x, y]], dtype=np.float32)
        input_label = np.array([1]) # foreground point

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        # choose the smallest mask
        mask_areas = masks.sum(axis=(1, 2))
        selected_idx = np.argmin(mask_areas)
        selected_mask = masks[selected_idx]

        # choose the highest scoring mask
        # selected_idx = np.argmax(scores)
        # selected_mask = masks[selected_idx]

        # check if the segment is disconnected:
        num_labels, labels = cv2.connectedComponents(selected_mask.astype(np.uint8))

        if num_labels > 2:
            continue  # skip disconnected masks


        # check if this leaf has already been segmented
        skip = False
        for existing_mask in leaf_masks:
            if mask_iou(selected_mask, existing_mask) > 0.3:
                skip = True
                break

        if skip:
            continue

        # remove large masks that could just be the background or entire plants
        if mask_areas[selected_idx] > 0.2 * height * width:
            continue

        # combine the masks
        combined_mask[selected_mask] = leaf_index
        leaf_index += 1

        # save the leaf mask
        leaf_masks.append(selected_mask)

    return combined_mask, leaf_masks


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union


# score the leaves, right now I am checking the disparity puts the target leaf closer to the camera than the border leaves
def score_leaves(depth_map, leaf_segmentations, border_width=2, disparity_threshold=0.01):
    
    scores = []
    
    for i in range(len(leaf_segmentations)):

        mask = leaf_segmentations[i].astype(np.uint8)
        # get leaf border
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        border = dilated - mask

        # get outer ring just beyond border
        outer = cv2.dilate(dilated, kernel, iterations=border_width) - dilated

        # coordinates
        border_coords = np.column_stack(np.where(border > 0))
        outer_coords = np.column_stack(np.where(outer > 0))

        if len(border_coords) == 0 or len(outer_coords) == 0:
            scores.append(0)
            continue

        num_occluded = 0
        num_checked = 0

        # for each border pixel, sample nearby outer pixels
        for (y, x) in border_coords:

            num_checked += 1

            y0, y1 = max(0, y - 2), min(depth_map.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(depth_map.shape[1], x + 3)

            border_depth = depth_map[y, x]
            local_outer = outer[y0:y1, x0:x1]
            outer_depths = depth_map[y0:y1, x0:x1][local_outer.astype(bool)]

            # check if the outer depths are closer to the camera (overlapping the target leaf)
            if np.any(outer_depths < border_depth - disparity_threshold): 
                num_occluded += 1

        scores.append(1 - (num_occluded / num_checked))

    return scores


def get_top_n_leaves(leaf_segmentations, scores, n=10):
    ranked_indices = np.argsort(scores)[::-1]

    leaves = []
    ranked_scores = []

    # get the top leaves/scores
    for i in range(min(n, len(leaf_segmentations))):
        idx = ranked_indices[i]
        leaves.append(leaf_segmentations[idx])
        ranked_scores.append(scores[idx])

    return leaves, ranked_scores

def load_sam(sam_path, sam_model_type):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


# save the segmentation mask as a png where the leaves are ranked by scores
def save_segmentation_mask(leaf_segmentations, scores, name, path, height, width):

    # change the labels based off the scores

    # the segmentation mask is stored where each leaf is labeled 1 - n

    # need to re-order these based off of the scores array

    # scores is an array where score for leaf labeled 1 is at index 0, score for leaf labeled 2 is at index 1 etc

    if leaf_segmentations is None or scores is None:
        save_empty_mask(height, width, name, path)
        return

    sorted_indices = np.argsort(scores)[::-1] # highest score first

    mapping = {}
    for new_label, idx in enumerate(sorted_indices, start=1):
        old_label = idx + 1
        mapping[old_label] = new_label

    remapped_mask = np.zeros_like(leaf_segmentations)

    for old_label, new_label in mapping.items():
        remapped_mask[leaf_segmentations == old_label] = new_label

    remapped_mask = remapped_mask.astype(np.uint8)

    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, name), remapped_mask)

def save_empty_mask(height, width, name, path):
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    print(empty_mask.shape)

    cv2.imwrite(os.path.join(path, name), empty_mask)


def main():
    data = load_data(IMAGE_DATA_DIR, DEPTH_DATA_DIR)

    # load the sam predictor
    sam_predictor = load_sam(SAM_MODEL_PATH, SAM_MODEL_TYPE) 

    show = False

    # for pair in data:
    for i in range(len(data)):

        image = data[i][0]
        depth_map = data[i][1]
        name = data[i][2]

        h, w = image.shape[:2]

        print(f"Processing image {name}")

        centroids = dbscan(depth_map, image, show=show)
        if len(centroids) == 0:
            save_segmentation_mask(None, None, name, OUTPUT_DIR, h, w)
            continue

        segmented_mask, leaf_segmentations = segment_with_sam(image, centroids, sam_predictor)

        if show:
            plot_segmentation_mask(image, segmented_mask)
        # plot_segmentation_mask(image, segmented_mask)

        scores = score_leaves(depth_map, leaf_segmentations, disparity_threshold=0.002, border_width=2)

        # ranked_leaves, scores = get_top_n_leaves(leaf_segmentations, scores)
        # print(scores)
        # visualise_top_leaves(image, ranked_leaves, scores)

        if show:
            visualise_top_leaves(image, leaf_segmentations, scores, n=5)

        # visualise_top_leaves(image, leaf_segmentations, scores, n=5)

        save_segmentation_mask(segmented_mask, scores, name, OUTPUT_DIR, h, w)


if __name__ == "__main__":
    main()
