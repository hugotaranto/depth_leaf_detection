# Image directory
# IMAGE_DIR = "../data/left"
IMAGE_DIR = "./data/images"

# manual annotation directory
ANNOTATION_DIR = "./data/annotation_out"

# SAM things
SAM_PATH = "./sam_checkpoints/sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE = "vit_l"

# Monocular depth estimation data
# depth type to be used in detection
DEPTH_TYPE = "MARIGOLD"

DEPTH_PRO_DIR = "./data/mono_depths/depth_pro"
MARIGOLD_DIR = "./data/mono_depths/marigold"

# leaf detection output
DETECTION_OUTPUT = "./data/detection_out"
# DETECTION_OUTPUT = "./data/samv3_out/merged"

DOWNSTREAM_DEPTH_TYPE = "DEPTH_PRO"

# downstream bins for scoring
BINS_FILE = "./bins.npz"
DATABASE = "./data/dataset.csv"

