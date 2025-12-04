import os

# 项目根目录（根据需要也可以直接写绝对路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMAGE_DIR = os.path.join(DATA_DIR, "raw")
PREPROCESSED_IMAGE_DIR = os.path.join(DATA_DIR, "preprocessed")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations")

# 确保目录存在
os.makedirs(RAW_IMAGE_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)
