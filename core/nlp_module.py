# core/nlp_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import json
import requests


@dataclass
class NLPConfig:
    # 目前只实现 ollama，本地跑 Qwen
    model_name: str = "qwen2.5:3b"
    api_url: str = "http://localhost:11434/api/generate"
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9


class NLPEngine:
    def __init__(self, config: Optional[NLPConfig] = None):
        self.config = config or NLPConfig()

    # ===== 对外三个接口 =====

    def summarize_text(self, text: str) -> str:
        """病历 / 报告摘要"""
        prompt = (
            "你是一名医疗 NLP 助手，请对下面的文本生成简短摘要，"
            "要求用中文，3~5 条要点，保留关键信息：\n\n"
            f"【原始文本】\n{text}\n\n"
            "【摘要要点】"
        )
        return self._generate(prompt)

    def analyze_medical_record(self, text: str) -> str:
        """病历解析：按照模板抽取结构化信息（但用纯文本输出）"""
        prompt = (
            "你是一名医疗 NLP 助手，请从下面的病历文本中抽取关键信息，"
            "并用中文按以下模板输出。如果某项缺失，就写“未知”。\n\n"
            f"【病历文本】\n{text}\n\n"
            "【输出模板】\n"
            "姓名：\n"
            "性别：\n"
            "年龄：\n"
            "主诉：\n"
            "现病史：\n"
            "既往史：\n"
            "检查所见：\n"
            "初步印象：\n"
            "建议检查：\n\n"
            "现在请严格按照上面的模板顺序输出结果："
        )
        return self._generate(prompt)

    def suggest_diagnosis(self, text: str) -> str:
        """诊断建议（课程设计演示用，不能当真实诊断）"""
        prompt = (
            "你是一名医学辅助决策系统，请根据下面的病历/检查描述，"
            "给出可能的诊断方向和进一步检查建议。注意：不能给出绝对诊断，"
            "只提供思路。\n\n"
            f"【病历/描述】\n{text}\n\n"
            "【输出要求】用中文分两部分回答：\n"
            "1. 可能的诊断方向（2~4 条）\n"
            "2. 建议进一步的检查或随访（2~4 条）\n\n"
            "最后请加上一句提示：本结果仅供教学与参考，不能作为正式医疗诊断依据。"
        )
        return self._generate(prompt)

    # ===== 内部：调用 Ollama =====

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,  # 一次性返回，简单一点
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
        }

        resp = requests.post(self.config.api_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # ollama 返回的主要内容在 "response"
        return data.get("response", "").strip()
