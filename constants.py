from depth_pro.depth_pro import DepthProConfig

# Directory to all plots
IMAGE_DIR = "./data/images"

# manual annotation directory
ANNOTATION_DIR = "./data/annotation_out"

# SAM things
SAM_PATH = "./sam_checkpoints/sam_vit_l_0b3195.pth"
SAM_MODEL_TYPE = "vit_l"

# Directory to save/load monocular depth estimations
DEPTH_PRO_DIR = "./data/mono_depths/depth_pro"
MARIGOLD_DIR = "./data/mono_depths/marigold"

# marigold model weights
MARIGOLD_CHECKPOINT = "./dependencies/Marigold/marigold-depth-v1-1"

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

# Depth map type to use for trait analysis
DOWNSTREAM_DEPTH_TYPE = "DEPTH_PRO"     # can be: DEPTH_PRO, MARIGOLD

# Database for manual scores of each plot
DATABASE = "./data/Trial 26.csv"

SAVOYNESS_EVAL_METHOD = "LAPLACE"       # can be: DEPTH, LAPLACE, FFT
CUPPING_EVAL_METHOD = "PLANE"           # can be: QUADRATIC, PLANE

# Directory to save savoyness scores to
SAVED_SAVOYNESS_SCORES = "./results/10_savoyness_depth_proposed.pkl"

# Directory to save cupping scores to
SAVED_CUPPING_SCORES = "./results/10_cupping_quadratic_proposed_44.pkl"
