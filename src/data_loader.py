# src/data_loader.py

import json
import logging
from typing import Generator, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_parse_jsonl(file_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    流式加载大型 JSONL 文件，提取标题和纯文本。

    Args:
        file_path (str): JSONL 文件的路径。

    Yields:
        tuple: (title: str, text_content: str)
    """
    logger.info(f"开始加载和解析 JSONL 文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                item = json.loads(line)

                title = item.get("title", "").strip()
                text_content = item.get("text", "").strip()

                # 可以根据需要添加过滤条件，例如过滤掉标题或内容过短的条目
                # if len(title) < 5 or len(text_content) < 50:
                #     logger.debug(f"跳过条目 (行 {line_num}): 标题或内容过短")
                #     continue

                if text_content: # 确保内容不为空
                    yield title, text_content
                else:
                    logger.debug(f"跳过条目 (行 {line_num}): 内容为空")

            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误 (行 {line_num}): {e}")
                logger.error(f"错误行内容: {line[:100]}...") # 打印错误行的前 100 个字符
                continue # 跳过这一行，继续处理下一行
            except Exception as e:
                logger.error(f"处理行 {line_num} 时发生未知错误: {e}")
                logger.error(f"错误行内容: {line[:100]}...")
                continue # 跳过这一行，继续处理下一行

    logger.info(f"成功解析完 JSONL 文件: {file_path}")

#if __name__ == "__main__":
    # 示例用法 (请替换为你的实际文件路径)
    #sonl_file_path = "../data/rag_docs.json" # 替换成实际路径
    #for title, content in load_and_parse_jsonl(jsonl_file_path):
    #     print(f"标题: {title}")
    #     print(f"内容预览: {content[:200]}...") # 打印前 200 个字符作为预览
    #     print("-" * 80)
    #     break # 只打印第一个条目作为示例
     # 主模块不执行任何操作，除非直接运行此脚本进行测试