import itertools
import pandas as pd

from validate import *
from plots import *

# This file contains methods used for producing plots and results seen in the report.

def validate_scoring():

    insets = [0, 1, 2, 4]
    border_distances = [0, 1, 2, 3, 4, 6]

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
                IMAGE_DIR,
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

def get_av_iou(csv_path):
    df = pd.read_csv(csv_path)

    methods = df["method"].unique()

    for method in methods:
        method_df = df[df["method"] == method]

        iou_scores = method_df["iou_mean"]
        # avg_iou = method_df["iou_mean"].mean()
        avg_iou = iou_scores.mean()

        print(f"{method}: {avg_iou:.4f}")


def generate_method_comparison_table(csv_path):

    import pandas as pd

    df = pd.read_csv(csv_path)

    methods = df["method"].unique()

    for method in methods:

        method_df = df[df["method"] == method]

        insets = sorted(method_df["inset"].unique())
        borders = sorted(method_df["border_distance"].unique())

        #
        # Find best score (for bolding)
        #

        best_idx = method_df["correct"].idxmax()

        best_inset = method_df.loc[best_idx, "inset"]
        best_border = method_df.loc[best_idx, "border_distance"]

        latex = []

        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append("\\renewcommand{\\arraystretch}{1.2}")
        latex.append("\\setlength{\\tabcolsep}{6pt}")
        latex.append("\\fontsize{10}{13}\\selectfont")

        #
        # Column formatting
        #
        # 2 left columns + inset columns
        #

        col_format = "c" * (len(insets) + 2)

        latex.append(f"\\begin{{tabular}}{{{col_format}}}")

        #
        # Top header row
        #

        latex.append(
            f"& & \\multicolumn{{{len(insets)}}}{{c}}{{\\textbf{{Inset}}}} \\\\"
        )

        #
        # Inset labels row
        #

        inset_header = "& \\multicolumn{1}{c|}{}"

        for inset in insets:
            inset_header += f" & \\textbf{{{inset}}}"

        inset_header += " \\\\"

        latex.append(inset_header)

        #
        # Partial horizontal line
        #

        latex.append(
            f"\\cline{{2-{len(insets)+2}}}"
        )

        #
        # Border distance label
        #

        latex.append(
            f"\\multirow{{{len(borders)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\textbf{{Border Distance}}}}}}"
        )

        #
        # Data rows
        #

        for i, border in enumerate(borders):

            row = []

            #
            # First column already occupied by multirow
            #

            row.append("&")

            #
            # Border label
            #

            row.append(
                f"\\multicolumn{{1}}{{c|}}{{\\textbf{{{border}}}}}"
            )

            #
            # Cells
            #

            for inset in insets:

                match = method_df[
                    (method_df["inset"] == inset) &
                    (method_df["border_distance"] == border)
                ]

                if len(match) == 0:
                    cell = "--"

                else:
                    correct = match.iloc[0]["correct"]
                    correct_requested = match.iloc[0]["correct_requested"]

                    #
                    # Bold best score
                    #

                    if inset == best_inset and border == best_border:

                        cell = (
                            f"\\textbf{{{correct:.1f}}}"
                            f"/"
                            f"\\textbf{{{correct_requested:.1f}}}"
                        )

                    else:

                        cell = (
                            f"{correct:.1f}"
                            f"/"
                            f"{correct_requested:.1f}"
                        )

                row.append(cell)

            #
            # Add line ending
            #

            # latex.append(" ".join(row) + " \\\\")
            latex.append(" & ".join(row) + " \\\\")

        latex.append("\\end{tabular}")

        #
        # Caption
        #

        latex.append(
            f"\\caption{{Validation results for {method}. Cells show "
            f"$\\frac{{\\text{{Correct}}}}{{\\text{{min(}}n\\text{{, no. detected leaves)}}}}$ "
            f"/ "
            f"$\\frac{{\\text{{Correct}}}}{{n}}$ "
            f"(\\%).}}"
        )

        latex.append(
            f"\\label{{tab:score_validation_{method}}}"
        )

        latex.append("\\end{table}")
        latex.append("")

        #
        # Print final table
        #

        print("\n".join(latex))
        print("\n\n")


def num_leaves_detected(mask_dir):


    masks = os.listdir(mask_dir)

    cum_leaves = 0
    count = 0

    for mask in masks:
        count += 1
        sys.stdout.write(f"Counting mask: {mask} || {count}\r")
        detections = load_image(mask, mask_dir)

        num_leaves = len(np.unique(detections)) - 1
        cum_leaves += num_leaves

    av_per_image = cum_leaves / count

    print("\n")
    print("Average leaves per image:", av_per_image)
    print("Num images:", count)

def test_num_leaves_used():

    num_leaves = [1, 2, 3, 5, 8, 10]

    detection_methods = {
        "proposed": DETECTION_OUTPUT,
        "rcnn": "./data/rcnn_detections",
        "yolo": "./data/yolo_detections",
        "samv3": "./data/samv3_out/merged"
    }

    params = {
        "proposed": (0, 1),
        "rcnn": (0, 6),
        "yolo": (0, 2),
        "samv3": (0, 4)
    }

    results = []

    for method_name in detection_methods.keys():
        detection_path = detection_methods[method_name]
        parameters = params[method_name]

        for n in num_leaves:
            metrics = validate_detection(
                IMAGE_DIR,
                num_leaves=n,
                annotation_dir=ANNOTATION_DIR,
                detection_output=detection_path,
                rescore=True,
                score_type="CUM",
                depth_type="DEPTH_PRO",
                inset=parameters[0],
                border_distance=parameters[1]
            )

            row = {
                "method": method_name,
                "number leaves": n,
                **metrics
            }

            results.append(row)

            print(
                f"number leaves = {n}, "
                f"correct = {metrics['correct']:.4f}"
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
    df.to_csv("./results/increased_score_val.csv", index=False)

if __name__ == "__main__":
    df = test_num_leaves_used()
    df.to_csv("./results/num_leaves.csv")

    plot_num_leaves_used(df, metric="correct")
    plot_num_leaves_used(df, metric="correct_requested", title="Requested Accuracy vs Number of Leaves Used")

    # main()
    

