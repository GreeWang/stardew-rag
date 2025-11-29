# src/generator.py

import logging
import os
from typing import List, Dict
from openai import OpenAI # 需要先 pip install openai

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGGenerator:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4",
        base_url: str = "https://bj.yi-zhan.top/v1",
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("必须提供 OpenAI API Key，或设置环境变量 OPENAI_API_KEY。")

        # 使用你的自定义 API 地址
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        logger.info(f"RAG Generator 使用模型: {model_name}, API地址: {base_url}")

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, str]], max_tokens: int = 512) -> str: # Add max_tokens parameter
        """
        根据查询和检索到的上下文生成答案。

        Args:
            query (str): 用户的查询问题。
            retrieved_chunks (List[Dict[str, str]]): 从检索器返回的 metadata 列表。
            max_tokens (int): LLM 生成的最大 token 数。

        Returns:
            str: LLM 生成的答案。
        """
        # 将检索到的块拼接成上下文字符串
        context_str = "\n\n".join([f"来源: {chunk['title']}\n内容: {chunk['text']}" for chunk in retrieved_chunks])

        # 设计 RAG Prompt
        # 这个 prompt 告诉 LLM 只能基于提供的上下文回答，并在不知道时明确说明
        prompt = f"""
        你是一个关于星露谷物语（Stardew Valley）的专家助手。
        请仅根据以下提供的上下文信息回答用户的问题。
        如果上下文信息不足以回答问题，请明确回答“根据提供的信息，我无法回答这个问题。”
        请确保你的回答准确、简洁，并引用信息的来源（标题）。

        上下文信息：
        {context_str}

        用户问题：
        {query}

        请给出你的回答：
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个 helpful 的星露谷物语知识助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens, # Use the passed max_tokens
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"调用 OpenAI API 时出错: {e}")
            return f"生成答案时发生错误: {e}"

# --- 如果你想使用本地模型 (例如通过 Ollama)，可以使用以下代码作为起点 ---
# import ollama
# class LocalRAGGenerator:
#     def __init__(self, model_name: str = "llama3"):
#         self.model_name = model_name
#         logger.info(f"Local RAG Generator 使用模型: {model_name}")
#
#     def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, str]]) -> str:
#         context_str = "\n\n".join([f"来源: {chunk['title']}\n内容: {chunk['text']}" for chunk in retrieved_chunks])
#         prompt = f"""
#         你是一个关于星露谷物语（Stardew Valley）的专家助手。
#         请仅根据以下提供的上下文信息回答用户的问题。
#         如果上下文信息不足以回答问题，请明确回答“根据提供的信息，我无法回答这个问题。”
#         请确保你的回答准确、简洁，并引用信息的来源（标题）。
#
#         上下文信息：
#         {context_str}
#
#         用户问题：
#         {query}
#
#         请给出你的回答：
#         """
#
#         try:
#             response = ollama.chat(
#                 model=self.model_name,
#                 messages=[{'role': 'user', 'content': prompt}],
#                 options={'temperature': 0.0} # 控制随机性
#             )
#             return response['message']['content'].strip()
#         except Exception as e:
#             logger.error(f"调用 Ollama API 时出错: {e}")
#             return f"生成答案时发生错误: {e}"

if __name__ == "__main__":
    # 示例用法 (需要设置 OPENAI_API_KEY 环境变量或在此处传入 key)
    # api_key = "your-openai-api-key-here"
    # generator = RAGGenerator(api_key=api_key)
    #
    # # 模拟检索结果 (实际使用时，这会来自 retriever.search)
    # mock_retrieved_chunks = [
    #     {"title": "枫糖浆 - 星露谷物语官方中文维基", "text": "来自Stardew Valley Wiki 枫糖浆 有着独特风味的甜浆。 信息 来源 打造 季节 任意季节 能量 / 生命值"},
    #     {"title": "树液采集器 - 星露谷物语官方中文维基", "text": "来自Stardew Valley Wiki 树液采集器 一种可以安装在枫树上以获取枫糖浆的设备。 信息 来源 打造 售出价格 250金 打造 配方来源 饮食 1 耐久度 用后即弃 所需材料 铜锭 （2） 木材 （30）"}
    # ]
    # query = "如何获得枫糖浆？"
    # answer = generator.generate_answer(query, mock_retrieved_chunks)
    # print(f"问题: {query}")
    # print(f"答案: {answer}")
    pass # 主模块不执行任何操作，除非直接运行此脚本进行测试
