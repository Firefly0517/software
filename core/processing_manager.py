import os
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from config import PREPROCESSED_IMAGE_DIR
from .io_utils import load_image


class ProcessingManager:
    """
    统一管理：
    - 图像加载
    - 各种预处理（降噪 / 裁剪 / 对齐 / 格式转换 / 旋转 / 翻转 / 直方图均衡）
    - 历史链 + 撤销 / 恢复
    - 每一步结果保存
    """

    def __init__(self):
        self.original_path: Optional[str] = None
        self.original_name: Optional[str] = None

        self._history: List[np.ndarray] = []
        self._history_tags: List[str] = []
        self._history_files: List[str] = []

        self._current_index: int = -1

        os.makedirs(PREPROCESSED_IMAGE_DIR, exist_ok=True)

    # ---------------- 基本信息 ----------------

    def has_image(self) -> bool:
        return self._current_index >= 0 and len(self._history) > 0

    def get_current_img(self) -> Optional[np.ndarray]:
        if not self.has_image():
            return None
        return self._history[self._current_index]

    def get_current_shape_str(self) -> str:
        img = self.get_current_img()
        if img is None:
            return "shape: N/A"
        return f"shape: {img.shape}"

    def get_current_display_name(self) -> str:
        if not self.has_image():
            return "无图像"

        base = f"{self.original_name}.png" if self.original_name else "unknown.png"
        tag = self._history_tags[self._current_index]
        if tag == "原图":
            return base
        return f"{base}（步骤: {tag}）"

    def get_current_saved_path(self) -> Optional[str]:
        if not self.has_image():
            return None
        file = self._history_files[self._current_index]
        if not file:
            return None
        return os.path.join(PREPROCESSED_IMAGE_DIR, file)

    def get_history_tags(self) -> List[str]:
        return list(self._history_tags)

    def get_current_index(self) -> int:
        return self._current_index

    # ---------------- 加载原图 ----------------

    def load_original(self, path: str) -> Dict[str, Any]:
        img = load_image(path)
        if img is None or img.size == 0:
            raise ValueError("加载到的图像为空")

        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        self.original_path = path
        self.original_name = name

        self._history = [img]
        self._history_tags = ["原图"]
        self._history_files = [""]
        self._current_index = 0

        return {
            "img": img,
            "shape": img.shape,
            "display_name": self.get_current_display_name(),
        }

    # ---------------- 撤销 / 恢复 ----------------

    def can_undo(self) -> bool:
        return self._current_index > 0

    def can_redo(self) -> bool:
        return self._current_index < len(self._history) - 1

    def undo(self) -> Optional[Dict[str, Any]]:
        if not self.can_undo():
            return None
        self._current_index -= 1
        img = self.get_current_img()
        return {
            "img": img,
            "shape": img.shape if img is not None else None,
            "display_name": self.get_current_display_name(),
        }

    def redo(self) -> Optional[Dict[str, Any]]:
        if not self.can_redo():
            return None
        self._current_index += 1
        img = self.get_current_img()
        return {
            "img": img,
            "shape": img.shape if img is not None else None,
            "display_name": self.get_current_display_name(),
        }

    # ---------------- 内部工具 ----------------

    def _save_step_image(self, img: np.ndarray, tag: str) -> str:
        if self.original_name is None:
            base_name = "image"
        else:
            base_name = self.original_name

        step_index = len(self._history)
        safe_tag = tag.replace(" ", "").replace("(", "").replace(")", "")
        filename = f"{base_name}_step{step_index}_{safe_tag}.png"
        save_path = os.path.join(PREPROCESSED_IMAGE_DIR, filename)
        cv2.imwrite(save_path, img)
        return filename

    def _push_history(self, img: np.ndarray, tag: str, saved_file: str):
        self._history = self._history[: self._current_index + 1]
        self._history_tags = self._history_tags[: self._current_index + 1]
        self._history_files = self._history_files[: self._current_index + 1]

        self._history.append(img)
        self._history_tags.append(tag)
        self._history_files.append(saved_file)
        self._current_index = len(self._history) - 1

    # ---------------- 降噪 ----------------

    def apply_denoise(
        self,
        method: str = "median",
        ksize: int = 3,
        sigmaX: float = 0.0,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
    ) -> Dict[str, Any]:
        img = self.get_current_img()
        if img is None:
            raise ValueError("当前没有可处理的图像")

        if ksize <= 0:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1

        if method == "median":
            out = cv2.medianBlur(img, ksize)
            tag = f"D_median{ksize}"
        elif method == "gaussian":
            out = cv2.GaussianBlur(img, (ksize, ksize), sigmaX if sigmaX > 0 else 0)
            tag = f"D_gauss{ksize}"
        elif method == "bilateral":
            out = cv2.bilateralFilter(
                img,
                d=bilateral_d,
                sigmaColor=bilateral_sigma_color,
                sigmaSpace=bilateral_sigma_space,
            )
            tag = f"D_bilat{bilateral_d}"
        else:
            raise ValueError(f"不支持的降噪方法: {method}")

        saved_file = self._save_step_image(out, tag)
        self._push_history(out, tag, saved_file)
        return {
            "img": out,
            "shape": out.shape,
            "display_name": self.get_current_display_name(),
            "saved_path": os.path.join(PREPROCESSED_IMAGE_DIR, saved_file),
            "tag": tag,
        }

    # ---------------- 裁剪 ----------------

    def apply_crop(self, x: int, y: int, w: int, h: int) -> Dict[str, Any]:
        img = self.get_current_img()
        if img is None:
            raise ValueError("当前没有可处理的图像")

        h_img, w_img = img.shape[:2]

        if w <= 0 or h <= 0:
            raise ValueError("裁剪宽高必须为正数")

        x = max(0, x)
        y = max(0, y)
        if x >= w_img or y >= h_img:
            raise ValueError("裁剪起点超出图像范围")

        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x or y2 <= y:
            raise ValueError("裁剪区域无效")

        cropped = img[y:y2, x:x2].copy()
        tag = f"Cr_{x}_{y}_{x2 - x}x{y2 - y}"

        saved_file = self._save_step_image(cropped, tag)
        self._push_history(cropped, tag, saved_file)

        return {
            "img": cropped,
            "shape": cropped.shape,
            "display_name": self.get_current_display_name(),
            "saved_path": os.path.join(PREPROCESSED_IMAGE_DIR, saved_file),
            "tag": tag,
        }

    # ---------------- 对齐（Align，简单平移示意） ----------------

    def apply_align(self, dx: int = 5, dy: int = 5) -> Dict[str, Any]:
        img = self.get_current_img()
        if img is None:
            raise ValueError("当前没有可处理的图像")

        h, w = img.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        tag = f"A_{dx}_{dy}"

        saved_file = self._save_step_image(out, tag)
        self._push_history(out, tag, saved_file)
        return {
            "img": out,
            "shape": out.shape,
            "display_name": self.get_current_display_name(),
            "saved_path": os.path.join(PREPROCESSED_IMAGE_DIR, saved_file),
            "tag": tag,
        }

    # ---------------- 格式转换（灰度 / RGB） ----------------

    def apply_color_convert(self, mode: str = "gray") -> Dict[str, Any]:
        img = self.get_current_img()
        if img is None:
            raise ValueError("当前没有可处理的图像")

        if mode == "gray":
            if img.ndim == 2:
                out = img.copy()
            else:
                out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tag = "C_toGray"
        elif mode == "rgb":
            if img.ndim == 3:
                out = img.copy()
            else:
                out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            tag = "C_toRGB"
        else:
            raise ValueError("不支持的格式转换模式")

        saved_file = self._save_step_image(out, tag)
        self._push_history(out, tag, saved_file)
        return {
            "img": out,
            "shape": out.shape,
            "display_name": self.get_current_display_name(),
            "saved_path": os.path.join(PREPROCESSED_IMAGE_DIR, saved_file),
            "tag": tag,
        }

    # ---------------- 旋转（90/180/270） ----------------

    def apply_rotate(self, angle: int = 90) -> Dict[str, Any]:
        img = self.get_current_img()
        if img is None:
            raise ValueError("当前没有可处理的图像")

        angle = angle % 360
        if angle == 90:
            out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            tag = "R_90"
        elif angle == 180:
            out = cv2.rotate(img, cv2.ROTATE_180)
            tag = "R_180"
        elif angle == 270:
            out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            tag = "R_270"
        else:
            raise ValueError("目前仅支持 90/180/270 度旋转")

        saved_file = self._save_step_image(out, tag)
        self._push_history(out, tag, saved_file)
        return {
            "img": out,
            "shape": out.shape,
            "display_name": self.get_current_display_name(),
            "saved_path": os.path.join(PREPROCESSED_IMAGE_DIR, saved_file),
            "tag": tag,
        }

    # ---------------- 翻转（水平 / 垂直） ----------------

    def apply_flip(self, mode: str = "h") -> Dict[str, Any]:
        img = self.get_current_img()
        if img is None:
            raise ValueError("当前没有可处理的图像")

        if mode == "h":
            out = cv2.flip(img, 1)
            tag = "F_h"
        elif mode == "v":
            out = cv2.flip(img, 0)
            tag = "F_v"
        else:
            raise ValueError("翻转模式仅支持 h / v")

        saved_file = self._save_step_image(out, tag)
        self._push_history(out, tag, saved_file)
        return {
            "img": out,
            "shape": out.shape,
            "display_name": self.get_current_display_name(),
            "saved_path": os.path.join(PREPROCESSED_IMAGE_DIR, saved_file),
            "tag": tag,
        }

    # ---------------- 直方图均衡 ----------------

    def apply_hist_equalize(self) -> Dict[str, Any]:
        img = self.get_current_img()
        if img is None:
            raise ValueError("当前没有可处理的图像")

        if img.ndim == 2:
            out = cv2.equalizeHist(img)
        else:
            # Y 通道均衡
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            ycrcb_eq = cv2.merge([y_eq, cr, cb])
            out = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

        tag = "HEq"
        saved_file = self._save_step_image(out, tag)
        self._push_history(out, tag, saved_file)
        return {
            "img": out,
            "shape": out.shape,
            "display_name": self.get_current_display_name(),
            "saved_path": os.path.join(PREPROCESSED_IMAGE_DIR, saved_file),
            "tag": tag,
        }

    def get_image_at_step(self, index: int) -> Optional[np.ndarray]:
        if 0 <= index < len(self._history):
            return self._history[index]
        return None

    def get_history_descriptions(self) -> List[str]:
        return self._history_tags.copy()
