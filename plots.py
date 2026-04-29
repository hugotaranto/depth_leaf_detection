import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import cv2

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
            color = (0, 0, 255)   # blue (matplotlib RGB later)
        else:
            color = (255, 0, 0)   # red

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
