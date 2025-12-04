import os
import cv2
import numpy as np

from config import PREPROCESSED_IMAGE_DIR

# 尝试 DICOM 支持
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def is_dicom_file(path: str) -> bool:
    """简单判断是否为 DICOM 文件"""
    if path.lower().endswith(".dcm"):
        return True
    try:
        with open(path, "rb") as f:
            header = f.read(132)
        return header[128:132] == b"DICM"
    except Exception:
        return False


def load_image(path: str) -> np.ndarray:
    """支持中文路径 + DICOM + 彩色图像"""

    # 1) DICOM
    if is_dicom_file(path) and HAS_PYDICOM:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max() * 255.0
        img = img.astype(np.uint8)

        # DICOM 通常是灰度，这里转成 3 通道，方便统一处理
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # 2) 普通图片：使用 imdecode 支持中文路径 + 保留颜色信息
    try:
        with open(path, "rb") as f:
            data = f.read()
        np_data = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)  # 保留原始通道

        if img is None:
            raise ValueError("OpenCV imdecode 返回 None")

        # 如果是灰度，保持 2D；如果是彩色，BGR 3 通道
        return img

    except Exception as e:
        raise RuntimeError(f"无法读取图像文件：{path}\n错误：{e}")


def save_processed_image(img: np.ndarray, original_path: str) -> str:
    """将处理后的影像保存到 data/preprocessed/"""
    os.makedirs(PREPROCESSED_IMAGE_DIR, exist_ok=True)

    base_name = os.path.basename(original_path)
    name, _ = os.path.splitext(base_name)

    save_name = f"{name}_processed.png"
    save_path = os.path.join(PREPROCESSED_IMAGE_DIR, save_name)

    # 使用 imwrite 保存
    cv2.imwrite(save_path, img)

    return save_path


def save_step_image(img: np.ndarray, base_name: str, step_name: str) -> str:
    """
    保存每一步处理结果。
    base_name: 原图文件名，如 image.png
    step_name: D / A / Cr / C
    """
    from config import PREPROCESSED_IMAGE_DIR
    os.makedirs(PREPROCESSED_IMAGE_DIR, exist_ok=True)

    name, _ = os.path.splitext(base_name)
    save_name = f"{name}_{step_name}.png"
    save_path = os.path.join(PREPROCESSED_IMAGE_DIR, save_name)
    cv2.imwrite(save_path, img)
    return save_path
