from typing import List, Dict


class AnnotationManager:
    """
    标注管理模块：
    - 课程设计中可以只用假数据模拟“影像标注”实体
    """

    def __init__(self):
        pass

    def auto_generate_dummy(self) -> List[Dict]:
        """
        自动生成一条简单的假标注数据，用于演示：
        影像标注 = 标注ID + 影像编号 + 病灶位置 + 病灶类型
        """
        annotations = []

        annotation = {
            "标注ID": "IMG000001_标注001",
            "影像编号": "IMG000001",
            "病灶位置": "x1=30,y1=30,x2=80,y2=80,层号=1",
            "病灶类型": "nodule"  # 结节
        }
        annotations.append(annotation)

        return annotations
