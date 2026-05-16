# call depth-pro and marigold from python script.
import numpy as np
import depth_pro as dp
import torch
import os
from util import *
import sys

from constants import DEPTH_PRO_CONFIG

def load_depth_pro(config=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config is None:
        model, transform = dp.create_model_and_transforms(device=device, precision=torch.half)
    else:
        model, transform = dp.create_model_and_transforms(config=config, device=device, precision=torch.half)

    model.eval()

    return model, transform

def depth_pro_predict(directory, model, transform, save_path=None):

    image_names = os.listdir(directory)

    depths = []

    for name in image_names:

        base_name = os.path.splitext(name)[0]
        saved = os.listdir(save_path)
        # check if the depth has already been calculated

        if save_path is not None and f"{base_name}.npy" in saved:
            # load it
            depth = load_std_depth(name, save_path)

        # otherwise compute it
        else:
            sys.stdout.write(f"Predicting image: {name}\r")
            # image = load_image(name, directory)
            image_path = os.path.join(directory, name)
            image, _, f_px = dp.load_rgb(image_path)

            image_t = transform(image)

            torch.cuda.reset_peak_memory_stats()
            prediction = model.infer(image_t, f_px=f_px)
            depth_tensor = prediction["depth"]

            depth = depth_tensor.detach().cpu().numpy()

            torch.cuda.empty_cache()

            if save_path is not None:
                # save the depth map to the save directory as a .npy
                os.makedirs(save_path, exist_ok=True)
                out_file = os.path.join(save_path, f"{base_name}.npy")
                np.save(out_file, depth)

        depths.append(depth)

    return np.array(depths)

def marigold_predict():
    pass


def main():

    image_dir = "./data/images_full"
    save_dir = "./data/mono_depths/depth_pro/full"

    dp_model, dp_transform = load_depth_pro(config=DEPTH_PRO_CONFIG)

    count = 0
    plots = os.listdir(image_dir)
    total = len(plots)

    for plot in plots:
        count += 1
        print(f"\nWorking on plot {plot}, {count}/{total}")

        plot_dir = os.path.join(image_dir, plot)
        cams = os.listdir(plot_dir)

        for cam in cams:
            cam_path = os.path.join(plot_dir, cam)

            # make the predictions
            depths = depth_pro_predict(cam_path, dp_model, dp_transform, save_dir)

if __name__ == "__main__":
    main()
