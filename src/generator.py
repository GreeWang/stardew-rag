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
        base_url: str = None,
        prompt_type: str = "standard",
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("必须提供 OpenAI API Key，或设置环境变量 OPENAI_API_KEY。")

        if not base_url:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.prompt_type = prompt_type  # 保存提示类型

        logger.info(f"RAG Generator 使用模型: {model_name}, API地址: {base_url}, 提示类型: {prompt_type}")

    def _build_standard_prompt(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        """标准提示模板"""
        context_str = "\n\n".join([
            f"来源: {chunk['title']}\n内容: {chunk['text']}"
            for chunk in retrieved_chunks
        ])

        prompt = f"""
你是一个关于星露谷物语（Stardew Valley）的专家助手。
请仅根据以下提供的上下文信息回答用户的问题。
如果上下文不足，请回答："根据提供的信息，我无法回答这个问题。"
回答需简洁并引用来源标题。

上下文：
{context_str}

用户问题：
{query}

请给出你的回答：
"""
        return prompt

    def _build_detailed_prompt(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        """详细提示模板 - 要求更全面的回答"""
        context_str = "\n\n".join([
            f"来源: {chunk['title']}\n内容: {chunk['text']}"
            for chunk in retrieved_chunks
        ])

        prompt = f"""
你是一个专业的星露谷物语游戏专家。请基于以下提供的游戏文档，详细且准确地回答用户的问题。

相关文档：
{context_str}

问题：{query}

请提供全面、准确且详细的回答，确保：
1. 涵盖所有关键信息
2. 如果文档中有具体数据或步骤，请准确引用
3. 回答要结构清晰，易于理解
4. 如果信息不足，请明确指出哪些方面缺乏信息

请给出你的回答：
"""
        return prompt

    def _build_stardew_specific_prompt(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        """星露谷专用提示模板 - 针对游戏特点优化"""
        context_str = "\n\n".join([
            f"来源: {chunk['title']}\n内容: {chunk['text']}"
            for chunk in retrieved_chunks
        ])

        prompt = f"""
你是一个资深的星露谷物语玩家和专家。请基于以下游戏文档，为玩家提供准确、实用的游戏信息。

游戏文档：
{context_str}

玩家问题：{query}

请以星露谷物语专家的身份回答，确保：
1. 信息准确且对玩家有帮助
2. 如果涉及游戏机制、获取方法或配方，请详细说明
3. 优先使用游戏内的术语和表达方式
4. 如果文档中有具体数值（如时间、价格、效果），请准确引用
5. 回答要实用，能帮助玩家解决问题或做出决策

请给出你的回答：
"""
        return prompt

    def _build_minimal_prompt(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        """最小提示模板 - 简洁版本"""
        context_str = "\n\n".join([
            f"{chunk['text']}"
            for chunk in retrieved_chunks
        ])

        prompt = f"""
基于以下信息回答问题：

{context_str}

问题：{query}

回答：
"""
        return prompt

    def _format_chunks(self, retrieved_chunks: List[Dict[str, str]]) -> str:
        """格式化检索到的块为字符串"""
        return "\n\n".join([
            f"来源: {chunk['title']}\n内容: {chunk['text']}"
            for chunk in retrieved_chunks
        ])

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
        # 根据提示类型选择不同的提示模板
        if self.prompt_type == "detailed":
            prompt = self._build_detailed_prompt(query, retrieved_chunks)
            system_message = "你是一个专业的星露谷物语游戏专家，能够提供详细准确的游戏信息。"
        elif self.prompt_type == "stardew_specific":
            prompt = self._build_stardew_specific_prompt(query, retrieved_chunks)
            system_message = "你是一个资深的星露谷物语玩家和专家，熟悉游戏的各个方面。"
        elif self.prompt_type == "minimal":
            prompt = self._build_minimal_prompt(query, retrieved_chunks)
            system_message = "你是一个有帮助的助手。"
        else:  # 默认使用标准提示
            prompt = self._build_standard_prompt(query, retrieved_chunks)
            system_message = "你是一个 helpful 的星露谷助手。"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
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