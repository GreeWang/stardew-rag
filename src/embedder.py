# src/embedder.py

import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化嵌入模型。

        Args:
            model_name (str): 要使用的 sentence-transformers 模型名称。
        """
        logger.info(f"正在加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"模型 {model_name} 加载完成。")

    def embed_chunks(self, chunks_file_path: str, output_embeddings_path: str, output_metadata_path: str):
        """
        从 JSONL 文件加载块，生成嵌入，并将嵌入和元数据分别保存到 .npy 和 .jsonl 文件。

        Args:
            chunks_file_path (str): 输入的 chunks JSONL 文件路径。
            output_embeddings_path (str): 输出的嵌入向量 .npy 文件路径。
            output_metadata_path (str): 输出的元数据 .jsonl 文件路径。
        """
        logger.info(f"开始加载分块数据: {chunks_file_path}")

        # 读取所有块
        titles = []
        texts = []
        with open(chunks_file_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                titles.append(item["title"])
                texts.append(item["text"])

        logger.info(f"共加载 {len(titles)} 个块。")

        logger.info("开始生成嵌入向量...")
        # 一次性生成所有文本的嵌入 (注意：如果文本量非常大，可能需要分批处理)
        embeddings = self.model.encode(texts, show_progress_bar=True)

        logger.info(f"嵌入向量形状: {embeddings.shape}")

        # 保存嵌入向量 (numpy array)
        logger.info(f"正在保存嵌入向量到: {output_embeddings_path}")
        np.save(output_embeddings_path, embeddings)

        # 保存元数据 (title, text)
        logger.info(f"正在保存元数据到: {output_metadata_path}")
        with open(output_metadata_path, 'w', encoding='utf-8') as f_out:
            for title, text in zip(titles, texts):
                metadata_obj = {
                    "title": title,
                    "text": text
                }
                f_out.write(json.dumps(metadata_obj, ensure_ascii=False) + '\n')

        logger.info("嵌入和元数据保存完成。")

    def embed_single_text(self, text: str) -> np.ndarray:
        """
        为单个文本生成嵌入向量。

        Args:
            text (str): 输入文本。

        Returns:
            np.ndarray: 嵌入向量。
        """
        return self.model.encode([text])[0] # 返回单个向量


if __name__ == "__main__":
    # --- 配置路径 ---
    INPUT_CHUNKS_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\chunks.jsonl"        # 你之前生成的分块文件
    OUTPUT_EMBEDDINGS_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\embeddings.npy" # 保存嵌入向量的路径
    OUTPUT_METADATA_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\chunk_metadata.jsonl" # 保存元数据的路径

    # --- 选择模型 ---
    # "all-MiniLM-L6-v2" 是一个轻量级、快速的模型，适合 CPU
    # "all-mpnet-base-v2" 或 "bge-small-en-v1.5" (或中文 "bge-small-zh-v1.5") 性能可能更好但稍慢
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # --- 执行嵌入 ---
    embedder = ChunkEmbedder(model_name=EMBEDDING_MODEL_NAME)
    embedder.embed_chunks(
        chunks_file_path=INPUT_CHUNKS_PATH,
        output_embeddings_path=OUTPUT_EMBEDDINGS_PATH,
        output_metadata_path=OUTPUT_METADATA_PATH
    )
