# src/processor.py

import os
import json
import logging
from typing import Generator, Tuple

from data_loader import load_and_parse_jsonl
from chunker import chunk_documents

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_and_save_chunks(
    input_file_path: str,
    output_file_path: str,
    chunk_size: int = 1024,
    overlap: int = 50
):
    """
    加载 JSONL 文件，将其分块，并将结果保存到新的 JSONL 文件。

    Args:
        input_file_path (str): 输入的 JSONL 文件路径。
        output_file_path (str): 输出的 JSONL 文件路径。
        chunk_size (int): 每个块的最大字符数。
        overlap (int): 相邻块之间的重叠字符数。
    """
    logger.info(f"开始处理文件: {input_file_path}")
    logger.info(f"分块参数: size={chunk_size}, overlap={overlap}")
    logger.info(f"输出文件: {output_file_path}")

    # 确保输出文件的目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")

    processed_count = 0
    total_chunks = 0

    # 读取原始文档
    documents = load_and_parse_jsonl(input_file_path)

    # 分块
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    # 保存分块结果
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for chunk_title, chunk_text in chunks:
            # 每个块保存为一个 JSON 对象，写入一行
            chunk_obj = {
                "title": chunk_title,
                "text": chunk_text
            }
            f_out.write(json.dumps(chunk_obj, ensure_ascii=False) + '\n')
            total_chunks += 1

            # 每处理 1000 个块，打印一次进度
            if total_chunks % 1000 == 0:
                logger.info(f"已处理 {total_chunks} 个块...")

    logger.info(f"处理完成！总共生成了 {total_chunks} 个块，已保存到 {output_file_path}")


if __name__ == "__main__":
    # --- 配置你的文件路径 ---
    # 请将这些路径修改为你的实际路径
    INPUT_JSONL_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\rag_docs.json"  # 你的原始 JSONL 文件路径
    OUTPUT_CHUNKS_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\chunks.jsonl" # 你想保存分块结果的路径

    # --- 配置分块参数 ---
    CHUNK_SIZE = 1024
    OVERLAP = 50

    # --- 执行处理 ---
    process_and_save_chunks(
        input_file_path=INPUT_JSONL_PATH,
        output_file_path=OUTPUT_CHUNKS_PATH,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

