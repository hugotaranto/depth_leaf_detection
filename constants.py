from depth_pro.depth_pro import DepthProConfig
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

DEPTH_PRO_DIR = "./data/mono_depths/depth_pro"
MARIGOLD_DIR = "./data/mono_depths/marigold"

# depth pro config
DEPTH_PRO_CONFIG = DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="./dependencies/ml-depth-pro/checkpoints/depth_pro.pt",
    decoder_features=256,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)

# leaf detection output
DETECTION_OUTPUT = "./data/detection_out"
# DETECTION_OUTPUT = "./data/samv3_out/merged"

DOWNSTREAM_DEPTH_TYPE = "DEPTH_PRO"

# downstream bins for scoring
BINS_FILE = "./bins.npz"
# DATABASE = "./data/dataset.csv"
DATABASE = "./data/Trial 26.csv"

SAVOYNESS_EVAL_METHOD = "FFT"       # can be: DEPTH, LAPLACE, FFT
CUPPING_EVAL_METHOD = "QUADRATIC"   # can be: QUADRATIC, PLANE

SAVED_SAVOYNESS_SCORES = "savoyness_fft.pkl"
# SAVED_SAVOYNESS_SCORES = None

SAVED_CUPPING_SCORES = "cupping.pkl"
# SAVED_CUPPING_SCORES = None

