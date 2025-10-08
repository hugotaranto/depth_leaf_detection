import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from segment_anything import sam_model_registry, SamPredictor
import torch

from plots import *

IMAGE_DATA_DIR = '../data/left'
DEPTH_DATA_DIR = '../Marigold/output/depth_npy'

DOWNSAMPLE_SIZE = 256

# base SAM model
SAM_MODEL_TYPE = 'vit_b'
SAM_MODEL_PATH = './sam_base_checkpoint.pth'

# load the data into a dictionary
def load_data(image_dir, depth_dir):

    # get all of the image names
    image_names = os.listdir(image_dir)

    image_depth_pairs = []

    for name in image_names:
        base_name = os.path.splitext(name)[0] # take the .png off

        # construct the depth name
        depth_name = f"{base_name}_depth.npy"

        # create the full paths
        image_path = os.path.join(image_dir, name)
        depth_path = os.path.join(depth_dir, depth_name)

        if not os.path.exists(depth_path):
            print(f"Could not find depth corresponding depth file: {depth_path}")
            continue

        # load the image and depth map
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)

        image_depth_pairs.append((image, depth))

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
def dbscan(depth_map, image):
    
    # resize the image -- DBSCAN struggles with native 1900 * 1900
    resized = cv2.resize(depth_map, (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE))

    # use kmeans to separate background (soil) from leaves
    mask = get_foreground_mask(resized)

    height, width = resized.shape
    yy, xx = np.mgrid[0:height, 0:width]

    # this is the coordinates of the points in 3d space
    coords = np.column_stack((xx.ravel(), yy.ravel(), resized.ravel()))
    filtered_points = coords[mask.ravel()].astype(np.float32)

    # scale the points
    scaled_points = StandardScaler().fit_transform(filtered_points)

    # DBSCAN parameters
    eps = 0.05
    min_samples = 20

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

    show_dbscan_clusters(resized, filtered_points[:, :2], labels, image, resized, centroids, orig_centroids)

    return centroids

# segment the leaves with sam
def segment_with_sam(image, centroids):
     
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    sam.to(device)
    predictor = SamPredictor(sam)

    predictor.set_image(image)

    height, width = image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    leaf_masks = []

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
        smallest_idx = np.argmin(mask_areas)
        smallest_mask = masks[smallest_idx]

        # remove large masks that could just be the background or entire plants
        if mask_areas[smallest_idx] > 0.2 * height * width:
            continue

        # combine the masks
        combined_mask[smallest_mask] = i

        # save the leaf mask
        leaf_masks.append(smallest_mask)

    return combined_mask, leaf_masks


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


def main():
    data = load_data(IMAGE_DATA_DIR, DEPTH_DATA_DIR)

    # for pair in data:
    for i in range(len(data)):

        image = data[i][0]
        depth_map = data[i][1]

        centroids = dbscan(depth_map, image)
        if len(centroids) == 0:
            continue

        segmented_mask, leaf_segmentations = segment_with_sam(image, centroids)
        plot_segmentation_mask(image, segmented_mask)

        scores = score_leaves(depth_map, leaf_segmentations)

        ranked_leaves, scores = get_top_n_leaves(leaf_segmentations, scores)
        print(scores)

        visualise_top_leaves(image, ranked_leaves, scores)


if __name__ == "__main__":
    main()
