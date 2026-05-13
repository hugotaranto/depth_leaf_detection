import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

DPI = 100

def display_depth(depth, cmap="plasma"):
    plt.imshow(depth, cmap=cmap)
    plt.axis("off")
    plt.show()

def plot_image_and_depth(image, depth, title=None, cmap="plasma"):
    """
    Display an RGB image and its depth map side by side.

    Parameters:
        image (np.ndarray): The RGB image (H, W, 3).
        depth (np.ndarray): The depth map (H, W).
        title (str, optional): Optional title for the figure.
        cmap (str, optional): Colormap for depth visualization (default: 'plasma').
    """
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
    """
    Plot depth map with DBSCAN clusters and centroids overlaid.

    Parameters:
        depth_map (np.ndarray): 2D depth map (H, W)
        labels_2d (np.ndarray): 2D array of cluster labels (-1 = noise)
        cluster_centroids (list of tuples): list of (x, y) coordinates for cluster centers
        cmap (str): colormap for depth map
    """
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


def show_dbscan_clusters(depth_map, filtered_xy, labels, image, depth, centroids, orig_centroids):
    """
    depth_map: 2D array (for grayscale display)
    filtered_xy: (N, 2) array of (x, y) coords used in clustering
    labels: DBSCAN cluster labels for each filtered point
    image: original RGB image (for context)
    """
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
    width, height = image.shape[:2]
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
    """
    image: original image (H, W, 3) in RGB
    pred: predicted mask (H, W)
    gt: ground truth mask (H, W)
    alpha: transparency of overlay
    """

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
    """
    mask: (H, W) boolean
    mono_depth: (H, W) depth map
    image: (H, W, 3) optional
    """
    
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
    """
    xs, ys, zs: leaf points
    a, b, c: plane coefficients (z = ax + by + c)
    """

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
    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.auto_scale_xyz(xs, ys, zs)

    fig.colorbar(scatter, ax=ax, label='Residual (cupping)')

    plt.tight_layout()
    plt.show()


def plot_leaf_quadratic(xs, ys, zs, coeffs_quad, image=None, mask=None):
    """
    xs, ys, zs: leaf points
    coeffs_quad: (qa, qb, qc, qd, qe, qf)
    """

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
    ax.invert_yaxis()
    ax.invert_zaxis()

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
    """
    Visualize cropped leaf scoring regions.

    Colors:
    - green = leaf mask
    - red   = inner border
    - blue  = outer ring
    """

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
    """
    Plot:
    - original leaf depth surface
    - smoothed leaf surface
    - residual colouring (savoyness texture)

    Parameters
    ----------
    xs, ys, zs:
        Original leaf points

    smooth_zs:
        Smoothed depth values at leaf points

    residuals:
        zs - smooth_zs

    image, mask:
        Optional RGB image + segmentation mask
    """

    #
    # Layout
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

        fig = plt.figure(figsize=(18, 8))

        #
        # image subplot
        #

        ax_img = fig.add_subplot(1, 2, 1)

        ax_img.imshow(image_crop)
        ax_img.set_title("Leaf")
        ax_img.axis("off")

        #
        # 3D subplot
        #

        ax = fig.add_subplot(1, 2, 2, projection='3d')

    else:

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    #
    # Original leaf surface
    #

    scatter = ax.scatter(
        xs,
        ys,
        zs,
        c=residuals,
        cmap='coolwarm',
        s=2,
        alpha=0.9,
        label='Original Surface'
    )

    #
    # Smoothed surface
    #

    ax.scatter(
        xs,
        ys,
        smooth_zs,
        c='black',
        s=1,
        alpha=0.15,
        label='Smoothed Surface'
    )

    #
    # Optional connecting lines
    # (shows residual displacement)
    #

    step = max(len(xs) // 1000, 1)

    for i in range(0, len(xs), step):

        ax.plot(
            [xs[i], xs[i]],
            [ys[i], ys[i]],
            [smooth_zs[i], zs[i]],
            alpha=0.08,
            linewidth=0.5,
            color='gray'
        )

    #
    # Formatting
    #

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Depth")

    ax.set_title("Leaf Savoyness")

    ax.view_init(elev=65, azim=90)

    ax.invert_yaxis()
    ax.invert_zaxis()

    ax.auto_scale_xyz(xs, ys, zs)

    fig.colorbar(
        scatter,
        ax=ax,
        label='Residual Texture'
    )

    plt.tight_layout()
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
    """
    Plot score distributions and learned thresholds.
    """

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


# def predict_image_scores(
#     test_leaf_scores,
#     bins,
#     method="binned_mean"
# ):
#     """
#     Generate image-level predictions.
#     """
#
#     preds = []
#
#     for scores in test_leaf_scores:
#
#         scores = np.asarray(scores)
#
#         if method == "raw_mean":
#
#             agg = np.mean(scores)
#             pred = np.digitize(agg, bins) + 1
#
#         elif method == "raw_median":
#
#             agg = np.median(scores)
#             pred = np.digitize(agg, bins) + 1
#
#         elif method == "binned_mean":
#
#             leaf_preds = np.digitize(scores, bins) + 1
#             pred = int(np.round(np.mean(leaf_preds)))
#
#         elif method == "binned_median":
#
#             leaf_preds = np.digitize(scores, bins) + 1
#             pred = int(np.round(np.median(leaf_preds)))
#
#         else:
#             raise ValueError("Unknown method")
#
#         preds.append(pred)
#
#     return np.asarray(preds)


def plot_prediction_scatter(
    gt,
    preds,
    title="Predicted vs Ground Truth"
):
    """
    Scatter plot of predictions vs GT.
    """

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
    """
    Plot confusion matrix.
    """

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
    """
    Visualize leaf score distributions by GT class.
    """

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
