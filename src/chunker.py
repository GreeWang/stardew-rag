# src/chunker.py

import logging
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_text(title: str, text: str, chunk_size: int = 1024, overlap: int = 50) -> List[Tuple[str, str]]:
    """
    将单个文档的标题和文本分割成块。

    Args:
        title (str): 文档标题。
        text (str): 文档文本。
        chunk_size (int): 每个块的最大字符数。默认为 1024。
        overlap (int): 相邻块之间的重叠字符数。默认为 50。

    Returns:
        List[Tuple[str, str]]: 包含 (chunk_title, chunk_text) 对的列表。
    """
    if not text:
        logger.warning(f"文档 '{title}' 的文本内容为空，无法分块。")
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # 如果当前块的起始位置加上块大小超过了文本总长度，则直接取到末尾
        if end >= text_length:
            chunk_text_content = text[start:]
        else:
            # 否则，取指定大小的块
            chunk_text_content = text[start:end]

        # 为每个块附加原始标题，方便后续索引和溯源
        chunks.append((title, chunk_text_content))

        # 计算下一个块的起始位置
        # 如果当前块已经到达或超过末尾，则结束循环
        if end >= text_length:
            break

        # 否则，移动到下一个块的起始位置，考虑重叠
        start = end - overlap

    logger.debug(f"文档 '{title}' 已被分割成 {len(chunks)} 个块。")
    return chunks

# --- 可选：批量处理多个文档 ---
def chunk_documents(documents: List[Tuple[str, str]], chunk_size: int = 1024, overlap: int = 50) -> List[Tuple[str, str]]:
    """
    批量处理多个 (title, text) 文档，将其分割成块。

    Args:
        documents (List[Tuple[str, str]]): 包含 (title, text) 对的列表。
        chunk_size (int): 每个块的最大字符数。默认为 1024。
        overlap (int): 相邻块之间的重叠字符数。默认为 50。

    Returns:
        List[Tuple[str, str]]: 包含所有分割后 (chunk_title, chunk_text) 对的列表。
    """
    all_chunks = []
    for title, text in documents:
        chunks = chunk_text(title, text, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks

if __name__ == "__main__":
    # 示例用法
    sample_title = "示例标题"
    sample_text = "这是一个很长的文本。" * 500  # 生成一个长文本进行测试

    print(f"原始文本长度: {len(sample_text)}")
    chunks = chunk_text(sample_title, sample_text, chunk_size=1024, overlap=50)

    print(f"分块数量: {len(chunks)}")
    for i, (chunk_title, chunk_content) in enumerate(chunks[:3]):  # 只打印前3个块作为示例
        print(f"--- 块 {i+1} ---")
        print(f"标题: {chunk_title}")
        print(f"内容长度: {len(chunk_content)}")
        print(f"内容预览: {chunk_content[:100]}...") # 打印前100个字符
        print("-" * 80)

    # 测试重叠效果 (使用更短的文本和块大小)
    short_text = "0123456789" * 15  # "0123456789" 重复 15 次，共 150 个字符
    short_chunks = chunk_text("短文本测试", short_text, chunk_size=50, overlap=10)
    print(f"\n短文本测试 (size=50, overlap=10):")
    for i, (chunk_title, chunk_content) in enumerate(short_chunks):
        print(f"  块 {i+1}: '{chunk_content}' (长度: {len(chunk_content)})")
        if i == 2: # 只看前几块
            break
