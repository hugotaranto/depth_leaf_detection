# Leaf Detection Pipeline

A leaf detection and segmentation pipeline combining modern foundation models for segmentation and monocular depth estimation.

## Installation

Create and activate a Conda environment:

```bash
conda create -n leaf_detect python=3.10
conda activate leaf_detect
```

Install project dependencies:

```bash
bash install.sh
```

---

## Required Models

### Segment Anything (SAM)

Repository:

https://github.com/facebookresearch/segment-anything

Download the required model checkpoint(s) from:

https://github.com/facebookresearch/segment-anything#model-checkpoints

---

### Depth Pro

Repository:

https://github.com/apple/ml-depth-pro

Follow the installation instructions provided in the repository.

---

### Marigold

Repository:

https://github.com/prs-eth/marigold

Download the depth estimation checkpoint from:

https://huggingface.co/prs-eth/marigold-depth-v1-1/tree/main

---

## Configuration

Before running the pipeline, update the paths and configuration variables in:

```python
constants.py
```

Ensure all model checkpoints, input directories, and output directories are correctly specified.

---

## Usage

An example workflow is provided in:

```python
main.py
```

Run the pipeline with:

```bash
python main.py
```

---

## Project Structure

```text
.
├── main.py                 # Example pipeline execution
├── constants.py            # Configuration and file paths
├── install.sh              # Dependency installation script
├── detect.py               # Leaf detection algorithms/pipeline
├── downstream.py           # Trait estimation/scoring algorithms
├── validate.py             # Classification of trait analysis/detection validation
├── annotate.py             # Manual annotation of leaf segments
├── util.py                 # General utility functions
├── report_validation.py    # Generation of results for report
└── plots.py                # All matplot functions for displaying figures
```

---

## Notes

- SAM checkpoints must be downloaded separately.
- Marigold checkpoints must be downloaded separately.
- Verify all paths in `constants.py` before execution.
- Example usage and expected outputs can be found in `main.py`.
