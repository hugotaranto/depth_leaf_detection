import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry, SamPredictor
import torch
from matplotlib.colors import ListedColormap

SAM_PATH = '../../y4/sem2/comp_vision/major/detection/sam_checkpoints/sam_vit_l_0b3195.pth'
SAM_MODEL_TYPE = 'vit_l'
IMAGE_DIR = '../data/left'
OUTPUT_DIR = './annotation_out'

def save_to_file(name, mask):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{name}")

    print(f"Saved annotations for image: {name} to {OUTPUT_DIR}")

    cv2.imwrite(save_path, mask.astype(np.uint8))

def segment_point(sam_predictor, point):

    masks, scores, _ = sam_predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True
    )

    mask_areas = masks.sum(axis=(1, 2))
    smallest_idx = np.argmin(mask_areas)
    smallest_mask = masks[smallest_idx]

    return smallest_mask

def load_sam(sam_path, sam_model_type):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

def interactive_hover(image, sam_predictor):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Hover to segment")

    h, w = image.shape[:2]

    # --- OVERLAYS ---
    combined_overlay = np.zeros((h, w, 4), dtype=np.float32)
    combined_display = ax.imshow(combined_overlay)

    hover_overlay = np.zeros((h, w, 4), dtype=np.float32)
    mask_display = ax.imshow(hover_overlay)

    # --- MASKS ---
    mask = np.zeros((h, w), dtype=np.uint8)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_num = 1

    # --- COLORING FUNCTION ---
    def mask_to_rgba(label_mask):
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        unique_labels = np.unique(label_mask)
        unique_labels = unique_labels[unique_labels != 0]

        safe_hues = np.concatenate([
            np.linspace(0, 70, 6),
            np.linspace(190, 300, 6),
        ])

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

        for label in unique_labels:
            hue = safe_hues[(label - 1) % len(safe_hues)]
            r, g, b = hsl_to_rgb(hue)
            rgba[label_mask == label] = [r, g, b, 0.6]

        return rgba

    # --- HOVER EVENT ---
    def on_move(event):
        nonlocal mask

        mask_display.set_data(np.zeros((h, w, 4), dtype=np.float32))

        if not event.inaxes:
            fig.canvas.draw_idle()
            return

        x, y = int(event.xdata), int(event.ydata)

        if x < 0 or y < 0 or x >= w or y >= h:
            return

        mask = segment_point(sam_predictor, [x, y])

        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[mask == 1] = [1.0, 0.0, 0.0, 0.5]

        mask_display.set_data(rgba)
        fig.canvas.draw_idle()

    # --- CLICK EVENT ---
    def onclick(event):
        nonlocal leaf_num, mask, combined_mask

        print("clicked")
        combined_mask[mask == 1] = leaf_num
        leaf_num += 1

        combined_display.set_data(mask_to_rgba(combined_mask))
        fig.canvas.draw_idle()

    # Bind events
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.show()

    return combined_mask

def main():

    sam_predictor = load_sam(SAM_PATH, SAM_MODEL_TYPE)

    image_names = os.listdir(IMAGE_DIR)

    for name in image_names:
        image_path = os.path.join(IMAGE_DIR, name)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        sam_predictor.set_image(image)

        # annotate_image(image)
        mask = interactive_hover(image, sam_predictor)

        save_to_file(name, mask)


if __name__ == "__main__":
    main()
