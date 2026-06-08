import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.stats import pearsonr

DPI = 100

def display_depth(depth, cmap="plasma"):
    plt.imshow(depth, cmap=cmap)
    plt.axis("off")
    plt.show()

def plot_image_and_depth(image, depth, title=None, cmap="plasma"):

    # ensure proper types
    image = np.asarray(image)
    depth = np.asarray(depth)

    # create the figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # plot RGB image
    axes[0].imshow(image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # plot depth map
    im = axes[1].imshow(depth, cmap=cmap)
    axes[1].set_title("Depth Map")
    axes[1].axis("off")

    # add colorbar for depth
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # optional overall title
    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_depth_with_clusters(depth_map, labels_2d, cluster_centroids, cmap='plasma'):

    plt.figure(figsize=(10, 8))
    
    # Show depth map
    plt.imshow(depth_map, cmap=cmap)
    
    # Overlay clusters (semi-transparent mask)
    mask = labels_2d > -1
    plt.imshow(mask, alpha=0.3, cmap='cool')  # highlight all detected clusters
    
    # Overlay centroids
    for x, y in cluster_centroids:
        plt.scatter(x, y, color='cyan', s=50, edgecolors='black', linewidth=1)
    
    plt.title(f"Detected {len(cluster_centroids)} clusters")
    plt.axis('off')
    plt.show()

def show_dbscan_pipeline(depth_map, image, filtered_xy, labels, centroids, downsample_size):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ------------------------------------------------------------------
    # Left: depth map
    # ------------------------------------------------------------------
    axes[0].imshow(depth_map, cmap="plasma")
    axes[0].axis("off")

    # ------------------------------------------------------------------
    # Right: RGB image with clusters + centroids
    # ------------------------------------------------------------------
    axes[1].imshow(image)
    axes[1].axis("off")

    # Scale cluster points from DBSCAN resolution back to image resolution
    img_h, img_w = image.shape[:2]

    scale_x = img_w / downsample_size
    scale_y = img_h / downsample_size

    cluster_xy = filtered_xy.astype(np.float32).copy()
    cluster_xy[:, 0] *= scale_x
    cluster_xy[:, 1] *= scale_y

    unique_labels = np.unique(labels)

    for label in unique_labels:

        if label == -1:
            continue
        else:
            color = plt.cm.tab10(label % 10)
            size = 8
            alpha = 0.7

        pts = cluster_xy[labels == label]

        axes[1].scatter(
            pts[:, 0],
            pts[:, 1],
            s=size,
            c=[color],
            alpha=alpha,
            edgecolors="none"
        )

    # Centroids (already in original image coordinates)
    if len(centroids) > 0:
        axes[1].scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="red",
            s=80,
            marker="x",
            linewidths=2
        )

    plt.tight_layout()
    plt.show()



def show_dbscan_clusters(depth_map, filtered_xy, labels, image, depth, centroids, orig_centroids):

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # --- Left: original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=50, marker='x')

    axes[2].imshow(depth, cmap="plasma")
    axes[2].axis("off")
    axes[2].set_title("Monocular Depth Map Estimation (Marigold)")

    # --- Right: clustered depth map
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])

    axes[1].imshow(depth_map, cmap='gray')
    axes[1].set_title(f"DBSCAN Clusters ({n_clusters} clusters)")
    axes[1].axis('off')

    for label in unique_labels:
        if label == -1:
            cluster_color = 'lightgray'
            size = 4
        else:
            cluster_color = plt.cm.tab10(label % 10)
            size = 15

        cluster_points = filtered_xy[labels == label]
        axes[1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=size,
            c=[cluster_color],
            alpha=0.7,
            edgecolors='none'
        )

    axes[1].scatter(orig_centroids[:, 0], orig_centroids[:, 1], c='red', s=50, marker='x')

    plt.tight_layout()
    plt.show()

def plot_segmentation_mask(image, mask):
    height, width = image.shape[:2]
    combined_mask = np.ma.masked_where(mask == 0, mask)  # mask out zeros

    # --- 1. Build custom "no-green" colormap ---
    # Use hues that avoid green (skip 90–170° range)
    safe_hues = np.concatenate([
        np.linspace(0, 70, 6),     # reds–oranges–yellows
        np.linspace(190, 300, 6),  # blues–purples–magentas
    ])

    # Number of unique nonzero labels in the mask
    num_labels = int(np.max(mask))
    hues = np.linspace(0, len(safe_hues) - 1, num_labels) % len(safe_hues)
    hues = safe_hues[hues.astype(int)]

    # Convert HSL to RGB
    def hsl_to_rgb(h, s=0.7, l=0.5):
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        if h < 60:      r, g, b = c, x, 0
        elif h < 120:   r, g, b = x, c, 0
        elif h < 180:   r, g, b = 0, c, x
        elif h < 240:   r, g, b = 0, x, c
        elif h < 300:   r, g, b = x, 0, c
        else:            r, g, b = c, 0, x
        return (r + m, g + m, b + m)

    rgb_colors = np.array([hsl_to_rgb(h) for h in hues])
    np.random.seed(42)
    np.random.shuffle(rgb_colors)  # mix up similar tones
    cmap = ListedColormap(rgb_colors)

    # --- 2. Plot ---
    fig = plt.figure(figsize=(width / (DPI * 2), height / (DPI * 2)), dpi=DPI)
    plt.imshow(image)
    plt.imshow(combined_mask, alpha=0.5, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def display_pred_vs_gt(image, pred, gt, alpha=0.5):

    # Ensure image is float in [0,1]
    img = image.copy().astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    # Create masks
    gt_mask = gt > 0
    pred_mask = pred > 0

    # Create overlay
    overlay = img.copy()

    # Regions
    gt_only = gt_mask & ~pred_mask
    pred_only = pred_mask & ~gt_mask
    overlap = gt_mask & pred_mask

    # Apply colours (RGB this time!)
    overlay[gt_only] = [0, 0, 1]     # blue
    overlay[pred_only] = [1, 0, 0]   # red
    overlay[overlap] = [1, 0, 1]     # purple

    # Blend
    blended = (1 - alpha) * img + alpha * overlay

    # Clip just in case
    blended = np.clip(blended, 0, 1)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.title("GT (blue) vs Pred (red) | Overlap (purple)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return blended


def visualise_top_leaves(image, leaf_segmentations, scores, n):
    vis = image.copy()

    total = len(scores)
    n = min(n, total)

    # --- sort by score (descending) ---
    sorted_idx = np.argsort(scores)[::-1]

    top_idx = set(sorted_idx[:n])
    all_idx = range(total)

    # --- Draw all leaves ---
    for i in all_idx:
        mask = leaf_segmentations[i].astype(np.uint8)
        score = scores[i]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # blue for top n, red otherwise
        if i in top_idx:
            color = (255, 0, 0)   # blue
        else:
            color = (0, 0, 255)   # red 

        cv2.drawContours(vis, contours, -1, color, 2)

        # centroid for label
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            rank = np.where(sorted_idx == i)[0][0] + 1
            label = f"#{rank}: {score:.2f}"

            cv2.putText(vis, label, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # --- Convert BGR → RGB for matplotlib ---
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # --- Display ---
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_rgb)
    plt.title(f"Top {n} (Blue) vs Others (Red)")
    plt.axis("off")
    plt.show()

def get_mask_contours(mask):
    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def draw_contour_overlay(image, mask, color=(255, 0, 0), thickness=1):
    overlay = image.copy()

    contours = get_mask_contours(mask)

    cv2.drawContours(
        overlay,
        contours,
        -1,
        color,
        thickness
    )

    return overlay

def plot_leaf_depth_3d(mask, mono_depth, downsample=1, image=None, disp_mask=None):
    
    if image is not None:

        if disp_mask is None:
            disp_mask = mask

        image = draw_contour_overlay(image, disp_mask)

        # now crop the image to the mask
        pad = 200
        ys, xs = np.where(disp_mask)

        if len(xs) == 0 or len(ys) == 0:
            return None  # empty mask

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # optional padding
        x_min = max(x_min - pad, 0)
        y_min = max(y_min - pad, 0)
        x_max = x_max + pad
        y_max = y_max + pad

        image = image[y_min:y_max, x_min:x_max]


    ys, xs = np.where(mask)
    zs = mono_depth[ys, xs]

    if downsample > 1:
        ys = ys[::downsample]
        xs = xs[::downsample]
        zs = zs[::downsample]

    # ---- figure layout ----
    if image is not None:
        fig = plt.figure(figsize=(16, 8))

        # Left: image
        ax_img = fig.add_subplot(1, 2, 1)
        ax_img.imshow(image)
        ax_img.set_title("Leaf (image)")
        ax_img.axis('off')

        # Right: 3D plot
        ax = fig.add_subplot(1, 2, 2, projection='3d')

    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # ---- 3D scatter ----
    scatter = ax.scatter(xs, ys, zs, c=zs, s=2)

    ax.view_init(elev=65, azim=90)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')


    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.auto_scale_xyz(xs, ys, zs)

    fig.colorbar(scatter, ax=ax, label='Depth')

    plt.tight_layout()
    plt.show()


def plot_leaf_from_points(xs, ys, zs, a, b, c, image=None, mask=None):

    # ---- compute plane + residuals ----
    z_plane = a * xs + b * ys + c
    residuals = zs - z_plane

    # ---- layout ----
    if image is not None and mask is not None:

        # make the cropped image with boundary around leaf
        image = draw_contour_overlay(image, mask)

        pad = 200
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        height, width, _ = image.shape

        x_min = max(x_min - pad, 0)
        y_min = max(y_min - pad, 0)
        x_max = min(x_max + pad, width)
        y_max = min(y_max + pad, height)

        image = image[y_min:y_max, x_min:x_max]

        fig = plt.figure(figsize=(16, 8))

        ax_img = fig.add_subplot(1, 2, 1)
        ax_img.imshow(image)
        ax_img.set_title("Leaf")
        ax_img.axis('off')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # ---- scatter coloured by residuals ----
    scatter = ax.scatter(xs, ys, zs, c=residuals, cmap='coolwarm', s=2)

    # ---- plane surface ----
    grid_x, grid_y = np.meshgrid(
        np.linspace(xs.min(), xs.max(), 30),
        np.linspace(ys.min(), ys.max(), 30)
    )
    grid_z = a * grid_x + b * grid_y + c

    ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.3, color='gray')

    # ---- formatting ----
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')

    ax.view_init(elev=65, azim=90)
    ax.invert_zaxis()
    ax.invert_xaxis()

    ax.auto_scale_xyz(xs, ys, zs)

    fig.colorbar(scatter, ax=ax, label='Residual (cupping)')

    plt.tight_layout()
    plt.show()


def plot_leaf_quadratic(xs, ys, zs, coeffs_quad, image=None, mask=None):

    qa, qb, qc, qd, qe, qf = coeffs_quad

    # ---- layout ----
    if image is not None and mask is not None:

        image = draw_contour_overlay(image, mask)

        pad = 200
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        height, width, _ = image.shape

        x_min = max(x_min - pad, 0)
        y_min = max(y_min - pad, 0)
        x_max = min(x_max + pad, width)
        y_max = min(y_max + pad, height)

        image = image[y_min:y_max, x_min:x_max]

        # shift coords to cropped frame
        xs_plot = xs - x_min
        ys_plot = ys - y_min

        fig = plt.figure(figsize=(16, 8))

        ax_img = fig.add_subplot(1, 2, 1)
        ax_img.imshow(image)
        ax_img.set_title("Leaf")
        ax_img.axis('off')

        ax = fig.add_subplot(1, 2, 2, projection='3d')

    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        xs_plot = xs
        ys_plot = ys

    # ---- scatter ----
    scatter = ax.scatter(xs_plot, ys_plot, zs, c=zs, s=2)

    # ---- quadratic surface ----
    grid_x, grid_y = np.meshgrid(
        np.linspace(xs.min(), xs.max(), 40),
        np.linspace(ys.min(), ys.max(), 40)
    )

    grid_z = (
        qa * grid_x**2 +
        qb * grid_y**2 +
        qc * grid_x * grid_y +
        qd * grid_x +
        qe * grid_y +
        qf
    )

    # shift grid if cropped
    if image is not None and mask is not None:
        grid_x_plot = grid_x - x_min
        grid_y_plot = grid_y - y_min
    else:
        grid_x_plot = grid_x
        grid_y_plot = grid_y

    ax.plot_surface(
        grid_x_plot,
        grid_y_plot,
        grid_z,
        alpha=0.4,
        color='green'
    )

    # ---- formatting ----
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')

    ax.view_init(elev=65, azim=90)
    ax.invert_zaxis()
    ax.invert_xaxis()

    ax.auto_scale_xyz(xs_plot, ys_plot, zs)

    fig.colorbar(scatter, ax=ax, label='Depth')

    plt.tight_layout()
    plt.show()

def visualise_leaf_regions(
    depth_map,
    mask,
    inner_border,
    outer_ring,
    padding=200,
    figsize=(8, 8),
):

    #
    # Find crop bounds from mask
    #

    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        print("Empty mask")
        return

    y0 = max(0, ys.min() - padding)
    y1 = min(depth_map.shape[0], ys.max() + padding)

    x0 = max(0, xs.min() - padding)
    x1 = min(depth_map.shape[1], xs.max() + padding)

    #
    # Crop everything
    #

    depth_crop = depth_map[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    inner_crop = inner_border[y0:y1, x0:x1]
    outer_crop = outer_ring[y0:y1, x0:x1]

    #
    # Normalize depth image
    #

    depth_vis = depth_crop.astype(np.float32).copy()

    valid = np.isfinite(depth_vis)

    if np.any(valid):
        dmin = np.min(depth_vis[valid])
        dmax = np.max(depth_vis[valid])

        if dmax > dmin:
            depth_vis[valid] = (
                (depth_vis[valid] - dmin)
                / (dmax - dmin)
            )

    #
    # Convert grayscale depth to RGB
    #

    rgb = np.stack([depth_vis] * 3, axis=-1)

    #
    # Overlay regions
    #

    rgb[mask_crop.astype(bool)] = [0.0, 1.0, 0.0]
    rgb[outer_crop.astype(bool)] = [0.0, 0.0, 1.0]
    rgb[inner_crop.astype(bool)] = [1.0, 0.0, 0.0]

    #
    # Display
    #

    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title("Leaf Scoring Regions")
    plt.axis("off")
    plt.show()


def plot_leaf_savoyness(
    xs,
    ys,
    zs,
    smooth_zs,
    residuals,
    image=None,
    mask=None,
):

    fig = plt.figure(figsize=(22, 8))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.5, 1.5], figure=fig)

    #
    # Optional RGB crop
    #

    if image is not None and mask is not None:

        image = draw_contour_overlay(image, mask)

        pad = 200

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        height, width = image.shape[:2]

        x_min = max(int(x_min - pad), 0)
        y_min = max(int(y_min - pad), 0)

        x_max = min(int(x_max + pad), width)
        y_max = min(int(y_max + pad), height)

        image_crop = image[y_min:y_max, x_min:x_max]

        # ax_img = fig.add_subplot(1, 3, 1)
        ax_img = fig.add_subplot(gs[0])

        ax_img.imshow(image_crop)
        ax_img.set_title("Leaf")
        ax_img.axis("off")

        # ax_orig = fig.add_subplot(1, 3, 2, projection='3d')
        ax_orig = fig.add_subplot(gs[1], projection='3d')
        # ax_smooth = fig.add_subplot(1, 3, 3, projection='3d')
        ax_smooth = fig.add_subplot(gs[2], projection='3d')

    else:

        ax_orig = fig.add_subplot(1, 2, 1, projection='3d')
        ax_smooth = fig.add_subplot(1, 2, 2, projection='3d')

    #
    # Original surface
    #

    ax_orig.scatter(
        xs,
        ys,
        zs,
        c=zs,
        cmap='viridis',
        s=2,
        alpha=0.9
    )

    ax_orig.set_title("Original Depth Surface")

    #
    # Smoothed surface
    #

    ax_smooth.scatter(
        xs,
        ys,
        smooth_zs,
        c=smooth_zs,
        cmap='viridis',
        s=2,
        alpha=0.9
    )

    ax_smooth.set_title("Smoothed Depth Surface")

    #
    # Shared formatting
    #

    z_min = min(zs.min(), smooth_zs.min())
    z_max = max(zs.max(), smooth_zs.max())

    for ax in [ax_orig, ax_smooth]:

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Depth")

        ax.view_init(elev=65, azim=90)

        ax.invert_xaxis()
        ax.invert_zaxis()

        ax.set_zlim(z_max, z_min)

        ax.auto_scale_xyz(xs, ys, zs)

    plt.tight_layout()
    plt.show()

def plot_selected_leaf(mask, image, im_size=500):

    #
    # Crop region around leaf
    #

    ys, xs = np.where(mask)

    y_size = np.max(ys) - np.min(ys)
    x_size = np.max(xs) - np.min(xs)

    y_padding = (im_size - y_size) // 2
    x_padding = (im_size - x_size) // 2

    y_min = max(int(ys.min() - y_padding), 0)
    y_max = min(int(ys.max() + y_padding), image.shape[0])

    x_min = max(int(xs.min() - x_padding), 0)
    x_max = min(int(xs.max() + x_padding), image.shape[1])

    image_crop = image[y_min:y_max, x_min:x_max]

    image_vis = draw_contour_overlay(
        image_crop.copy(),
        mask[y_min:y_max, x_min:x_max]
    )

    fig = plt.figure(frameon=False)

    plt.axis("off")
    plt.imshow(image_vis)

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    plt.margins(0, 0)

    cv2.imwrite("savoyness_gt3.png", image_vis)

    plt.show()


def plot_savoyness_process(mask, image, texture, valid_mask, clean_mask, lap, crop_padding=200):

    #
    # Crop region around leaf
    #

    ys, xs = np.where(mask)

    y_min = max(int(ys.min() - crop_padding), 0)
    y_max = min(int(ys.max() + crop_padding), image.shape[0])

    x_min = max(int(xs.min() - crop_padding), 0)
    x_max = min(int(xs.max() + crop_padding), image.shape[1])

    #
    # Cropped images
    #

    image_crop = image[y_min:y_max, x_min:x_max]

    texture_crop = texture[y_min:y_max, x_min:x_max]

    lap_crop = lap[y_min:y_max, x_min:x_max]

    valid_crop = valid_mask[y_min:y_max, x_min:x_max]

    #
    # Overlay visualisation
    #

    overlay = np.zeros_like(lap_crop)

    overlay[valid_crop] = np.abs(
        lap_crop[valid_crop]
    )

    #
    # Draw contour overlay
    #

    image_vis = draw_contour_overlay(
        image_crop.copy(),
        mask[y_min:y_max, x_min:x_max]
    )

    removed_pixels = (
        clean_mask &
        ~valid_mask
    )

    removed_vis = image_crop.copy()

    removed_crop = removed_pixels[
        y_min:y_max,
        x_min:x_max
    ]

    #
    # colour removed pixels red
    #

    removed_vis[removed_crop] = [255, 0, 0]

    #
    # Plot
    #

    fig, axs = plt.subplots(
        1,
        5,
        figsize=(24, 6)
    )

    axs[0].imshow(image_vis)
    axs[0].set_title("Leaf")

    axs[1].imshow(removed_vis)
    axs[1].set_title("Removed Bright Pixels")

    axs[2].imshow(
        texture_crop,
        cmap='gray'
    )
    axs[2].set_title("High-pass Texture")

    axs[3].imshow(
        lap_crop,
        cmap='inferno'
    )
    axs[3].set_title("Laplacian")

    axs[4].imshow(
        overlay,
        cmap='inferno'
    )
    axs[4].set_title("Savoy Texture")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_fft_savoyness_process(
    image,
    mask,
    texture,
    patch_center,
    patch_size,
    magnitude,
    freq_mask,
    crop_padding=200
):

    x, y = patch_center
    half = patch_size // 2

    #
    # Crop around leaf
    #

    ys, xs = np.where(mask)

    y_min = max(int(ys.min() - crop_padding), 0)
    y_max = min(int(ys.max() + crop_padding), image.shape[0])

    x_min = max(int(xs.min() - crop_padding), 0)
    x_max = min(int(xs.max() + crop_padding), image.shape[1])

    image_crop = image[y_min:y_max, x_min:x_max]
    texture_crop = texture[y_min:y_max, x_min:x_max]

    #
    # Draw contour
    #

    image_vis = draw_contour_overlay(
        image_crop.copy(),
        mask[y_min:y_max, x_min:x_max]
    )

    #
    # Draw FFT patch location
    #

    rect_x = x - half - x_min
    rect_y = y - half - y_min

    rect = plt.Rectangle(
        (rect_x, rect_y),
        patch_size,
        patch_size,
        edgecolor='red',
        facecolor='none',
        linewidth=2
    )

    #
    # Extract patch
    #

    patch = texture[
        y-half:y+half,
        x-half:x+half
    ]

    #
    # Frequency band overlay
    #

    fft_overlay = magnitude.copy()

    fft_overlay[~freq_mask] *= 0.15

    #
    # Plot
    #

    fig, axs = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    #
    # Leaf image
    #

    axs[0].imshow(image_vis)
    axs[0].add_patch(rect)
    axs[0].set_title("Leaf + Sampled Patch")

    #
    # Texture image
    #

    axs[1].imshow(
        texture_crop,
        cmap='gray'
    )

    axs[1].add_patch(
        plt.Rectangle(
            (rect_x, rect_y),
            patch_size,
            patch_size,
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
    )

    axs[1].set_title("High-pass Texture")

    #
    # FFT magnitude
    #

    axs[2].imshow(
        magnitude,
        cmap='inferno'
    )

    axs[2].set_title("FFT Magnitude Spectrum")

    #
    # Frequency band
    #

    axs[3].imshow(
        fft_overlay,
        cmap='inferno'
    )

    axs[3].set_title("Selected Frequency Band")

    #
    # Formatting
    #

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_savoyness_process_grid(
    mask,
    image,
    texture,
    valid_mask,
    clean_mask,
    lap,
    crop_padding=200
):

    #
    # Crop region around leaf
    #

    ys, xs = np.where(mask)

    y_min = max(int(ys.min() - crop_padding), 0)
    y_max = min(int(ys.max() + crop_padding), image.shape[0])

    x_min = max(int(xs.min() - crop_padding), 0)
    x_max = min(int(xs.max() + crop_padding), image.shape[1])

    #
    # Cropped images
    #

    image_crop = image[y_min:y_max, x_min:x_max]

    texture_crop = texture[y_min:y_max, x_min:x_max]

    lap_crop = lap[y_min:y_max, x_min:x_max]

    #
    # Draw contour overlay
    #

    image_vis = draw_contour_overlay(
        image_crop.copy(),
        mask[y_min:y_max, x_min:x_max]
    )

    #
    # Removed bright pixels visualisation
    #

    removed_pixels = (
        clean_mask &
        ~valid_mask
    )

    removed_crop = removed_pixels[
        y_min:y_max,
        x_min:x_max
    ]

    removed_vis = image_crop.copy()

    #
    # Colour removed pixels red
    #

    removed_vis[removed_crop] = [255, 0, 0]

    #
    # Plot layout
    #

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(12, 12)
    )

    #
    # Top-left
    #

    axs[0, 0].imshow(image_vis)
    axs[0, 0].set_title("Segmented Leaf")

    #
    # Top-right
    #

    axs[0, 1].imshow(removed_vis)
    axs[0, 1].set_title("Removed Bright Pixels")

    #
    # Bottom-left
    #

    axs[1, 0].imshow(
        texture_crop,
        cmap='gray'
    )
    axs[1, 0].set_title("LAB Leaf Texture")

    #
    # Bottom-right
    #

    axs[1, 1].imshow(
        lap_crop,
        cmap='inferno'
    )
    axs[1, 1].set_title("Laplacian")

    #
    # Formatting
    #

    for row in axs:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_strategy_comparison(
    results_med,
    results_mean,
    results_all,
    title="Savoyness Strategy Comparison"
):
    """
    Compare evaluation metrics across fitting strategies.
    """

    strategies = [
        ("Fit Medians", results_med),
        ("Fit Means", results_mean),
        ("Fit All Leaves", results_all)
    ]

    metric_names = ["mae", "accuracy", "off_by_one"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric in zip(axs, metric_names):

        labels = []
        vals = []

        for fit_name, result_dict in strategies:

            for method_name, metrics in result_dict.items():

                labels.append(f"{fit_name}\n{method_name}")
                vals.append(metrics[metric])

        ax.bar(range(len(vals)), vals)

        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.set_title(metric.upper())
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_bins(
    train_scores,
    train_labels,
    bins,
    title="Learned Thresholds"
):

    plt.figure(figsize=(10, 6))

    classes = sorted(np.unique(train_labels))

    for cls in classes:

        vals = [
            s for s, l in zip(train_scores, train_labels)
            if l == cls
        ]

        plt.hist(
            vals,
            bins=20,
            alpha=0.5,
            label=f"Class {cls}"
        )

    for b in bins:
        plt.axvline(
            b,
            linestyle='--',
            linewidth=2
        )

    plt.xlabel("Continuous Score")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()

    plt.show()


def plot_prediction_scatter(
    gt,
    preds,
    title="Predicted vs Ground Truth"
):

    gt = np.asarray(gt)
    preds = np.asarray(preds)

    plt.figure(figsize=(6, 6))

    plt.scatter(gt, preds, alpha=0.7)

    mn = min(gt.min(), preds.min())
    mx = max(gt.max(), preds.max())

    plt.plot([mn, mx], [mn, mx], '--')

    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title)

    plt.grid(True)

    plt.show()


def plot_confusion(
    gt,
    preds,
    n_classes=9,
    title="Confusion Matrix"
):

    cm = confusion_matrix(
        gt,
        preds,
        labels=np.arange(1, n_classes + 1)
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(cm)

    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")

    plt.xticks(range(n_classes), range(1, n_classes + 1))
    plt.yticks(range(n_classes), range(1, n_classes + 1))

    plt.title(title)

    plt.show()


def plot_leaf_score_distributions(
    test_leaf_scores,
    test_labels,
    title="Leaf Score Distributions"
):

    plt.figure(figsize=(10, 6))

    classes = sorted(np.unique(test_labels))

    for cls in classes:

        vals = []

        for scores, label in zip(
            test_leaf_scores,
            test_labels
        ):

            if label == cls:
                vals.extend(scores)

        plt.hist(
            vals,
            bins=20,
            alpha=0.5,
            label=f"GT {cls}"
        )

    plt.xlabel("Leaf Score")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.legend()

    plt.show()


def visualise_leaf_pairing(
    depth_map,
    mask,
    inner_border,
    outer_ring,
    inner_point,
    outer_point,
    score,
    padding=200,
    figsize=(8, 8),
):

    #
    # Crop bounds
    #

    ys, xs = np.where(mask > 0)

    if len(ys) == 0:
        print("Empty mask")
        return

    y0 = max(0, ys.min() - padding)
    y1 = min(depth_map.shape[0], ys.max() + padding)

    x0 = max(0, xs.min() - padding)
    x1 = min(depth_map.shape[1], xs.max() + padding)

    #
    # Crop maps
    #

    depth_crop = depth_map[y0:y1, x0:x1]

    mask_crop = mask[y0:y1, x0:x1]
    inner_crop = inner_border[y0:y1, x0:x1]
    outer_crop = outer_ring[y0:y1, x0:x1]

    #
    # Normalise depth
    #

    depth_vis = depth_crop.astype(np.float32).copy()

    valid = np.isfinite(depth_vis)

    if np.any(valid):

        dmin = np.min(depth_vis[valid])
        dmax = np.max(depth_vis[valid])

        if dmax > dmin:

            depth_vis[valid] = (
                (depth_vis[valid] - dmin)
                / (dmax - dmin)
            )

    #
    # RGB image
    #

    rgb = np.stack(
        [depth_vis] * 3,
        axis=-1
    )

    #
    # Overlay masks
    #

    rgb[mask_crop.astype(bool)] = [0.0, 1.0, 0.0]
    rgb[outer_crop.astype(bool)] = [0.0, 0.0, 1.0]
    rgb[inner_crop.astype(bool)] = [1.0, 0.0, 0.0]

    #
    # Convert points into crop coordinates
    #

    iy, ix = inner_point
    oy, ox = outer_point

    iy_c = iy - y0
    ix_c = ix - x0

    oy_c = oy - y0
    ox_c = ox - x0

    #
    # Plot
    #

    plt.figure(figsize=figsize)

    plt.imshow(rgb)

    #
    # connection line
    #

    plt.plot(
        [ix_c, ox_c],
        [iy_c, oy_c],
        linewidth=2
    )

    #
    # inner point
    #

    plt.scatter(
        ix_c,
        iy_c,
        s=120,
        marker='o',
        edgecolors='black',
        linewidths=2,
        label='Inner Border'
    )

    #
    # outer point
    #

    plt.scatter(
        ox_c,
        oy_c,
        s=120,
        marker='x',
        linewidths=3,
        label='Outer Ring'
    )

    plt.title(
        f"Depth Difference: {score:.4f}"
    )

    plt.legend()

    plt.axis("off")

    plt.show()

    #
    # Print info
    #

    print("INNER POINT:")
    print(f"  y={iy}, x={ix}")
    print(f"  depth={depth_map[iy, ix]:.4f}")

    print()

    print("OUTER POINT:")
    print(f"  y={oy}, x={ox}")
    print(f"  depth={depth_map[oy, ox]:.4f}")

    print()

    print(f"DIFFERENCE: {score:.4f}")


def plot_sam_segmentation(image, mask, padding=200):

    # mask coordinates
    ys, xs = np.where(mask)

    x = int(xs.mean())
    y = int(ys.mean())

    if len(xs) == 0 or len(ys) == 0:
        print("Empty mask provided.")
        return

    # mask bounding box size
    mask_w = xs.max() - xs.min()
    mask_h = ys.max() - ys.min()

    # crop size based on bbox + padding
    crop_w = mask_w + 2 * padding
    crop_h = mask_h + 2 * padding

    # center crop on point prompt
    x_min = max(x - crop_w // 2, 0)
    x_max = min(x + crop_w // 2, image.shape[1])

    y_min = max(y - crop_h // 2, 0)
    y_max = min(y + crop_h // 2, image.shape[0])

    # crop image + mask
    cropped_img = image[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # point in crop coordinates
    crop_x = x - x_min
    crop_y = y - y_min

    # bbox in crop coordinates
    bx0 = xs.min() - x_min
    bx1 = xs.max() - x_min
    by0 = ys.min() - y_min
    by1 = ys.max() - y_min

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ---------------------------------------------------
    # Left: point prompt
    # ---------------------------------------------------
    axes[0].imshow(cropped_img)
    axes[0].scatter(crop_x, crop_y, c='red', s=80)
    axes[0].set_title("Leaf Centroid")
    axes[0].axis("off")

    # ---------------------------------------------------
    # Right: segmentation + bounding box
    # ---------------------------------------------------
    overlay = cropped_img.copy()

    # overlay[cropped_mask.astype(bool)] = (
    #     0.6 * overlay[cropped_mask.astype(bool)] +
    #     0.4 * np.array([0, 255, 0])
    # ).astype(np.uint8)

    axes[1].imshow(overlay)

    rect = plt.Rectangle(
        (bx0, by0),
        bx1 - bx0,
        by1 - by0,
        edgecolor='red',
        facecolor='none',
        linewidth=2
    )

    axes[1].add_patch(rect)

    axes[1].set_title("Leaf Bounding Box")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_num_leaves_used(df, metric="correct", title="Accuracy vs Number of Leaves Used"):

    plt.figure(figsize=(8, 5))

    methods = df["method"].unique()

    for method in methods:

        method_df = (
            df[df["method"] == method]
            .sort_values("number leaves")
        )

        x = method_df["number leaves"]
        y = method_df[metric]

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=method.upper()
        )

    plt.xlabel("Number of Leaves Used")
    plt.ylabel("Accuracy (%)")
    plt.title(title)

    plt.xticks(sorted(df["number leaves"].unique()))

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_gt_coverage(
    ground_truth,
    num_classes,
    score_type="Score"
):

    #
    # Compute counts
    #

    classes = np.arange(1, num_classes + 1)

    unique, counts = np.unique(ground_truth, return_counts=True)

    count_dict = dict(zip(unique, counts))

    frequencies = np.array([
        count_dict.get(cls, 0)
        for cls in classes
    ])

    percentages = (
        frequencies / np.sum(frequencies)
    ) * 100

    #
    # Create dataframe
    #

    df = pd.DataFrame({
        "Class": classes,
        "Count": frequencies,
        "Percentage": percentages
    })

    #
    # Plot
    #

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        classes,
        frequencies
    )

    max_height = np.max(frequencies)
    plt.ylim(0, max_height * 1.15)

    #
    # Add labels above bars
    #

    for bar, count, pct in zip(bars, frequencies, percentages):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=14
        )

    plt.xlabel(f"{score_type} Class")
    plt.ylabel("Number of Samples")

    plt.title(
        f"{score_type} Ground Truth Distribution"
    )

    plt.xticks(classes)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    #
    # Print LaTeX table
    #

    total_count = np.sum(frequencies)

    latex = []

    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\renewcommand{\\arraystretch}{1.2}")
    latex.append("\\setlength{\\tabcolsep}{10pt}")

    latex.append("\\begin{tabular}{ccc}")
    latex.append("\\hline")

    latex.append(
        "\\textbf{Class} & "
        "\\textbf{Count} & "
        "\\textbf{Percentage (\\%)} \\\\"
    )

    latex.append("\\hline")

    for cls, count, pct in zip(classes, frequencies, percentages):

        latex.append(
            f"{cls} & "
            f"{count} & "
            f"{pct:.1f} \\\\"
        )

    latex.append("\\hline")

    #
    # Total row
    #

    latex.append(
        f"\\textbf{{Total}} & "
        f"\\textbf{{{total_count}}} & "
        f"\\textbf{{100.0}} \\\\"
    )

    latex.append("\\hline")
    latex.append("\\end{tabular}")

    latex.append(
        f"\\caption{{Distribution of ground truth "
        f"{score_type.lower()} scores across all classes.}}"
    )

    latex.append(
        f"\\label{{tab:{score_type.lower()}_gt_distribution}}"
    )

    latex.append("\\end{table}")

    print("\n".join(latex))

    return df



def plot_best_strategy_results(
    results,
    fit_strategy,
    prediction_strategy,
    title="Savoyness",
    n_classes=9
):

    data = results[fit_strategy][prediction_strategy]

    preds = data["preds"]
    labels = data["labels"]

    #
    # Correlation statistics
    #

    r, p = pearsonr(labels, preds)

    print(f"\nFit strategy: {fit_strategy}")
    print(f"Prediction strategy: {prediction_strategy}")

    print(f"Pearson r = {r:.4f}")
    print(f"p-value = {p:.6f}")

    print(f"MAE = {np.mean(data['mae']):.4f}")
    print(f"Accuracy = {np.mean(data['accuracy']):.4f}")
    print(f"Off-by-one = {np.mean(data['off_by_one']):.4f}")

    #
    # Scatter plot
    #

    plot_prediction_scatter(
        labels,
        preds,
        title=(
            f"{title} Predictions\n"
            f"{fit_strategy} + {prediction_strategy}"
        )
    )

    #
    # Confusion matrix
    #

    plot_confusion(
        labels,
        preds,
        n_classes=n_classes,
        title=(
            f"{title} Confusion Matrix\n"
            f"{fit_strategy} + {prediction_strategy}"
        )
    )

def plot_cv_metrics(
    results,
    title="Cross Validation Strategy Comparison",
    n_leaves=None
):

    metrics = [
        "mae",
        "accuracy",
        "off_by_one"
    ]

    strategy_labels = []

    metric_means = {
        m: [] for m in metrics
    }

    metric_stds = {
        m: [] for m in metrics
    }

    #
    # Flatten results hierarchy
    #

    for fit_name, fit_results in results.items():

        for pred_method, vals in fit_results.items():

            label = (
                f"{fit_name.replace('_', ' ').title()}\n"
                f"{pred_method}"
            )

            strategy_labels.append(label)

            for metric in metrics:

                metric_means[metric].append(
                    np.mean(vals[metric])
                )

                metric_stds[metric].append(
                    np.std(vals[metric])
                )

    #
    # Plot
    #

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(22, 6)
    )

    for ax, metric in zip(axs, metrics):

        ax.bar(
            range(len(strategy_labels)),
            metric_means[metric],
            yerr=metric_stds[metric]
        )

        ax.set_xticks(
            range(len(strategy_labels))
        )

        ax.set_xticklabels(
            strategy_labels,
            rotation=55,
            ha='right',
            fontsize=10
        )

        ax.set_title(
            metric.upper(),
            fontsize=16
        )

        ax.grid(
            True,
            alpha=0.3
        )

    full_title = title

    if n_leaves is not None:
        full_title += f" ({n_leaves} leaves)"

    fig.suptitle(
        full_title,
        fontsize=18
    )

    plt.tight_layout()
    plt.show()


def plot_fit_strategy_metrics(
    results,
    title="Fit Strategy Comparison",
    n_leaves=None
):

    metrics = [
        "mae",
        "accuracy",
        "off_by_one"
    ]

    fit_strategies = list(results.keys())

    metric_means = {
        m: [] for m in metrics
    }

    metric_stds = {
        m: [] for m in metrics
    }

    #
    # Average over prediction methods
    #

    for fit_name in fit_strategies:

        fit_results = results[fit_name]

        for metric in metrics:

            vals = []

            for pred_method in fit_results:

                vals.extend(
                    fit_results[pred_method][metric]
                )

            metric_means[metric].append(
                np.mean(vals)
            )

            metric_stds[metric].append(
                np.std(vals)
            )

    #
    # Plot
    #

    x = np.arange(len(fit_strategies))

    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in metrics:

        ax.plot(
            x,
            metric_means[metric],
            marker='o',
            linewidth=2,
            label=metric.upper()
        )

        ax.errorbar(
            x,
            metric_means[metric],
            yerr=metric_stds[metric],
            fmt='none',
            capsize=4
        )

    labels = [
        s.replace("_", " ").title()
        for s in fit_strategies
    ]

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=13
    )

    ax.set_ylabel(
        "Score",
        fontsize=15
    )

    full_title = title

    if n_leaves is not None:
        full_title += f" ({n_leaves} leaves)"

    ax.set_title(
        full_title,
        fontsize=18
    )

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_prediction_strategy_metrics(
    results,
    fit_strategy,
    title=None,
    n_leaves=None
):

    metrics = [
        "mae",
        "accuracy",
        "off_by_one"
    ]

    prediction_methods = [
        "raw_mean",
        "raw_median",
        "binned_mean",
        "binned_median"
    ]

    metric_means = {
        m: [] for m in metrics
    }

    metric_stds = {
        m: [] for m in metrics
    }

    #
    # Compute means/stds
    #

    fit_results = results[fit_strategy]

    for pred_method in prediction_methods:

        vals = fit_results[pred_method]

        for metric in metrics:

            metric_means[metric].append(
                np.mean(vals[metric])
            )

            metric_stds[metric].append(
                np.std(vals[metric])
            )

    #
    # Plot
    #

    x = np.arange(len(prediction_methods))

    fig, ax = plt.subplots(figsize=(11, 6))

    for metric in metrics:

        ax.plot(
            x,
            metric_means[metric],
            marker='o',
            linewidth=2,
            label=metric.upper()
        )

        ax.errorbar(
            x,
            metric_means[metric],
            yerr=metric_stds[metric],
            fmt='none',
            capsize=4
        )

    labels = [
        s.replace("_", "\n")
        for s in prediction_methods
    ]

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=12
    )

    ax.set_ylabel(
        "Score",
        fontsize=15
    )

    if title is None:

        title = (
            fit_strategy.replace("_", " ").title()
            + " Prediction Strategy Comparison"
        )

    if n_leaves is not None:
        title += f" ({n_leaves} leaves)"

    ax.set_title(
        title,
        fontsize=18
    )

    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def print_strategy_latex_table(
    all_results,
    caption="Classification strategy comparison.",
    label="tab:strategy_comparison"
):

    fit_names = {
        "fit_medians": "Medians",
        "fit_means": "Means",
        "fit_all_leaves": "All Leaves"
    }

    pred_names = {
        "raw_mean": "Raw Mean",
        "raw_median": "Raw Median",
        "binned_mean": "Binned Mean",
        "binned_median": "Binned Median"
    }

    latex = []

    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\renewcommand{\\arraystretch}{1.2}")
    latex.append("\\setlength{\\tabcolsep}{5pt}")
    latex.append("\\fontsize{8.5}{10}\\selectfont")

    latex.append("\\begin{tabular}{cllcccc}")
    latex.append("\\hline")

    latex.append(
        " & "
        "\\textbf{Fit} & "
        "\\textbf{Prediction} & "
        "\\textbf{MAE} & "
        "\\textbf{Acc.} & "
        "\\textbf{Off-One} & "
        "\\textbf{$r$} \\\\"
    )

    latex.append("\\hline")

    #
    # Each scoring method
    #

    for method_name, results in all_results.items():

        first_method_row = True

        for fit_key in results.keys():

            first_fit_row = True

            for pred_key in results[fit_key].keys():

                data = results[fit_key][pred_key]

                mae = np.mean(data["mae"])
                acc = np.mean(data["accuracy"]) * 100
                off1 = np.mean(data["off_by_one"]) * 100

                preds = np.asarray(data["preds"])
                labels = np.asarray(data["labels"])

                r, _ = pearsonr(labels, preds)

                #
                # Vertical method column
                #

                if first_method_row:

                    row = (
                        f"\\multirow{{12}}{{*}}{{"
                        f"\\rotatebox[origin=c]{{90}}{{{method_name}}}"
                        f"}}"
                    )

                    first_method_row = False

                else:

                    row = ""

                #
                # Fit strategy column
                #

                if first_fit_row:

                    row += (
                        f" & \\multirow{{4}}{{*}}{{{fit_names[fit_key]}}}"
                    )

                    first_fit_row = False

                else:

                    row += " & "

                #
                # Prediction strategy
                #

                row += (
                    f" & {pred_names[pred_key]}"
                    f" & {mae:.3f}"
                    f" & {acc:.1f}"
                    f" & {off1:.1f}"
                    f" & {r:.3f} \\\\"
                )

                latex.append(row)

            latex.append("\\cline{2-7}")

        latex.append("\\hline")

    latex.append("\\end{tabular}")

    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")

    latex.append("\\end{table}")

    # print("\\n".join(latex))
    print("\n".join(latex))




def plot_leaf_count_comparison(
    leaf_num_results,
    title="Leaf Count Comparison",
    fitting_method="fit_means",
    prediction_method="binned_mean",
    metric="mae",
    ylabel="Mean Absolute Error (MAE)"
):

    leaf_counts = sorted(leaf_num_results.keys())
    methods = sorted(leaf_num_results[leaf_counts[0]].keys())

    results_arrs = {}
    for method in methods:
        results_arrs[method] = []

    for n in leaf_counts:

        results = leaf_num_results[n]

        for method in methods:
            scores = results[method][fitting_method][prediction_method][metric]

            mean = np.mean(scores)
            results_arrs[method].append(mean)

    #
    # Plot
    #

    plt.figure(figsize=(7, 5))

    for method in methods:
        plt.plot(
            leaf_counts,
            results_arrs[method],
            marker='o',
            label=method.split(" ")[0]
        )

    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)

    plt.xlabel("Number of Leaves", fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    plt.title(title)

    plt.grid(True, alpha=0.3)

    plt.xticks(leaf_counts)

    plt.legend()

    plt.tight_layout()

    plt.show()


def plot_background_red(image, mask):

    plt.figure(figsize=(12, 6))

    # Red overlay where mask == 0
    overlay = image.copy()

    red_mask = mask == 0

    # Blend red into masked regions
    overlay[red_mask] = (
        0.5 * overlay[red_mask] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    plt.axis("off")
    plt.title("Background Mask Removed", fontsize=20)
    plt.imshow(overlay)

    plt.show()


def plot_segs_depth_scores(
    image,
    segmentation_mask,
    depth_map,
    scores,
):

    for i in range(len(scores)):
        scores[i] = scores[i] * 10

    def get_leaf_colors(num_labels):

        safe_hues = np.concatenate([
            np.linspace(0, 70, 6),
            np.linspace(190, 300, 6),
        ])

        hues = np.linspace(
            0,
            len(safe_hues) - 1,
            num_labels
        ) % len(safe_hues)

        hues = safe_hues[hues.astype(int)]

        def hsl_to_rgb(h, s=0.95, l=0.5):

            c = (1 - abs(2 * l - 1)) * s
            x = c * (1 - abs((h / 60) % 2 - 1))
            m = l - c / 2

            if h < 60:
                r, g, b = c, x, 0
            elif h < 120:
                r, g, b = x, c, 0
            elif h < 180:
                r, g, b = 0, c, x
            elif h < 240:
                r, g, b = 0, x, c
            elif h < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            return (r + m, g + m, b + m)

        colors = np.array([
            hsl_to_rgb(h)
            for h in hues
        ])

        np.random.seed(42)
        np.random.shuffle(colors)

        return colors

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7)
    )

    #
    # Generate colours once so both plots match
    #

    unique_ids = np.unique(segmentation_mask)
    unique_ids = unique_ids[unique_ids > 0]

    colors_float = get_leaf_colors(len(unique_ids))
    colors_uint8 = (colors_float * 255).astype(np.uint8)

    #
    # =====================================================
    # 1. RGB + segmentation overlay
    # =====================================================
    #

    combined_mask = np.ma.masked_where(
        segmentation_mask == 0,
        segmentation_mask
    )

    seg_cmap = ListedColormap(colors_float)

    axes[0].imshow(image)

    axes[0].imshow(
        combined_mask,
        alpha=0.5,
        cmap=seg_cmap,
        vmin=1,
        vmax=len(unique_ids)
    )

    # axes[0].set_title("Segmentation Mask")
    axes[0].axis("off")

    #
    # =====================================================
    # 2. Depth map
    # =====================================================
    #

    im = axes[1].imshow(
        depth_map,
        cmap="plasma"
    )

    # axes[1].set_title("Depth Map")
    axes[1].axis("off")

    # fig.colorbar(
    #     im,
    #     ax=axes[1],
    #     fraction=0.046,
    #     pad=0.04
    # )

    #
    # =====================================================
    # 3. Leaf scores
    # =====================================================
    #

    vis = image.copy()

    for idx, seg_id in enumerate(unique_ids):

        mask = (
            segmentation_mask == seg_id
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        color_rgb = colors_uint8[idx]

        color_bgr = (
            int(color_rgb[2]),
            int(color_rgb[1]),
            int(color_rgb[0]),
        )

        cv2.drawContours(
            vis,
            contours,
            -1,
            color_bgr,
            3
        )

        M = cv2.moments(mask)

        if M["m00"] > 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            score = scores[idx]

            cv2.putText(
                vis,
                f"{score:.2f}",
                (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_bgr,
                2,
                cv2.LINE_AA
            )

    axes[2].imshow(vis)
    # axes[2].set_title("Leaf Scores")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
