import cv2
import numpy as np
import time


class Preprocessor:
    """可配置的链式影像预处理"""

    def __init__(self):
        pass

    def denoise(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        return cv2.medianBlur(img, 3)

    def align(self, img: np.ndarray) -> np.ndarray:
        # 占位功能
        return img

    def crop(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return img  # 避免裁剪导致图像空尺寸而崩溃
        return img[h // 10: h * 9 // 10, w // 10: w * 9 // 10]

    def convert_format(self, img: np.ndarray) -> np.ndarray:
        return img

    def run(
        self,
        img: np.ndarray,
        do_denoise=False,
        do_align=False,
        do_crop=False,
        do_convert=False
    ) -> np.ndarray:

        if img is None or img.size == 0:
            raise ValueError("预处理收到的图像为空，无法进行处理")

        out = img

        if do_denoise:
            out = self.denoise(out)

        if do_align:
            out = self.align(out)

        if do_crop:
            out = self.crop(out)

        if do_convert:
            out = self.convert_format(out)

        return out
