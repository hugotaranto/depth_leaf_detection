import itertools
import pandas as pd

from validate import *

def validate_scoring():

    # insets = [0, 1, 2, 4, 8, 12]
    # border_distances = [0, 1, 2, 4, 8, 12]
    insets = [4]
    border_distances = [4]

    detection_methods = {
        "proposed": DETECTION_OUTPUT,
        "rcnn": "./data/rcnn_detections",
        "yolo": "./data/yolo_detections",
        "samv3": "./data/samv3_out/merged"
    }

    results = []

    for method_name, detection_path in detection_methods.items():

        print(f"\n=== Testing {method_name} ===")

        for inset, border_distance in itertools.product(insets, border_distances):

            metrics = validate_detection(
                "../data/left",
                num_leaves=5,
                annotation_dir=ANNOTATION_DIR,
                detection_output=detection_path,
                rescore=True,
                score_type="CUM",
                depth_type="DEPTH_PRO",
                inset=inset,
                border_distance=border_distance
            )

            row = {
                "method": method_name,
                "inset": inset,
                "border_distance": border_distance,
                **metrics
            }

            results.append(row)

            print(
                f"inset={inset:2d}, "
                f"border={border_distance:2d}, "
                f"correct={metrics['correct']:.4f}, "
                f"iou={metrics['iou_mean']:.4f}"
            )

    df = pd.DataFrame(results)

    return df


def main():
    # -=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-==-=-=-=-=-=-=-
    # Validating the scoring for the report

    df = validate_scoring()

    print("\nBest IOU:")
    print(df.sort_values("iou_mean", ascending=False).head())

    print("\nBest Detection Accuracy:")
    print(df.sort_values("correct", ascending=False).head())

    print("\nSaving to CSV: ./results/score_val.csv")
    df.to_csv("./results/score_val.csv", index=False)

if __name__ == "__main__":
    main()
