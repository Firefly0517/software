import numpy as np
from typing import Dict, List, Any


class SimpleAIDiagnosis:
    """
    简化版 AI 诊断模块：
    - 课程设计中可以只实现一个非常简单的规则/占位算法
    - 重点突出“预留 AI 模块接口”
    """

    def __init__(self):
        pass

    def predict(self, img: np.ndarray, metadata: Dict[str, Any], annotations: List[Dict]) -> Dict[str, Any]:
        """
        简单规则：
        - 如果存在标注，则判断为“可疑病灶”
        - 否则判断为“未见明显异常”
        """
        if annotations:
            diagnosis = "可疑病灶，建议进一步检查"
        else:
            diagnosis = "未见明显异常"

        result = {
            "诊断结论": diagnosis,
            "影像编号": metadata.get("影像编号", "未知"),
            "检查部位": metadata.get("检查部位", "other"),
        }
        return result
