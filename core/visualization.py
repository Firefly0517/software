import cv2
import numpy as np
from typing import Optional


def show_image_window(title: str, img: np.ndarray):
    """
    使用 OpenCV 弹出窗口显示图像。
    注意：在某些环境（如 PyCharm/远程服务器）可能不方便，
    但课程设计报告中可以写“支持简单可视化功能”。
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_with_text_overlay(img: np.ndarray, text: str, save_path: str):
    """
    保存带有文字标注的图像。
    可以在报告中展示“处理结果截图”。
    """
    output = img.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        output,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(save_path, output)
