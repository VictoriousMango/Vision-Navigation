# Visual SLAM Configuration
# This file contains example paths for KITTI dataset

# Example KITTI dataset paths (modify according to your setup)
KITTI_SEQUENCES_PATH = "data/sequences"
KITTI_POSES_PATH = "data/poses"

# Example for sequence 00
SEQUENCE_00_IMAGES = "data/sequences/00/image_0"
SEQUENCE_00_CALIB = "data/sequences/00/calib.txt"
SEQUENCE_00_POSES = "data/poses/00.txt"

# SLAM Parameters
DEFAULT_FEATURE_DETECTOR = "FD_SIFT"
DEFAULT_FEATURE_MATCHER = "FM_BF_NORM_L2"
DEFAULT_OPTIMIZATION_INTERVAL = 5
DEFAULT_SEQUENCE_LENGTH = 100

# Processing Options
ENABLE_LOOP_DETECTION = True
ENABLE_OPTIMIZATION = True
ENABLE_VISUALIZATION = True
