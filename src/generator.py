# src/generator.py

import logging
import os
from typing import List, Dict
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGGenerator:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4",
        base_url: str = "https://api.chsdw.top/v1"
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("必须提供 OpenAI API Key，或设置环境变量 OPENAI_API_KEY。")

        # 使用你的自定义 API 地址
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        logger.info(f"RAG Generator 使用模型: {model_name}, API地址: {base_url}")

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:

        context_str = "\n\n".join([
            f"来源: {chunk['title']}\n内容: {chunk['text']}"
            for chunk in retrieved_chunks
        ])

        prompt = f"""
你是一个关于星露谷物语（Stardew Valley）的专家助手。
请仅根据以下提供的上下文信息回答用户的问题。
如果上下文不足，请回答：“根据提供的信息，我无法回答这个问题。”
回答需简洁并引用来源标题。

上下文：
{context_str}

用户问题：
{query}

请给出你的回答：
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个 helpful 的星露谷助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=512
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"调用 API 出错: {e}")
            return f"生成答案时发生错误: {e}"


if __name__ == "__main__":
    pass

