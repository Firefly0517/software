import numpy as np
from typing import Dict, Any

# 如果需要用到 DICOM 的真实元数据，可以扩展这里
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def parse_metadata(image: np.ndarray, dicom_dataset: "pydicom.dataset.FileDataset" = None) -> Dict[str, Any]:
    """
    根据课程设计中的“影像元数据”定义，构造一个简单的元数据字典：
    影像元数据 = 影像编号 + 层数 + 像素间距 + 检查部位

    这里做一个简化实现：
    - 影像编号：IMG000001 这种形式
    - 层数：默认为 1
    - 像素间距：如果有 DICOM，就用 PixelSpacing，否则用 (1.0, 1.0)
    - 检查部位：先写 'other'
    """
    h, w = image.shape[:2]
    metadata = {}

    # 简单编号可后续改进
    metadata["影像编号"] = "IMG000001"

    # 层数（课程设计简化）
    metadata["层数"] = 1

    # 像素间距
    if dicom_dataset is not None and HAS_PYDICOM:
        spacing = getattr(dicom_dataset, "PixelSpacing", [1.0, 1.0])
        try:
            px, py = float(spacing[0]), float(spacing[1])
        except Exception:
            px, py = 1.0, 1.0
    else:
        px, py = 1.0, 1.0

    metadata["像素间距"] = (px, py)

    # 检查部位（这里写死为 other，报告里可写“当前版本仅支持通用影像”）
    metadata["检查部位"] = "other"

    # 额外补充一些信息（非文档必须）
    metadata["宽度"] = w
    metadata["高度"] = h

    return metadata
