# src/retriever.py

import json
import logging
import numpy as np
import faiss
import os
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorRetriever:
    def __init__(self, embeddings_path: str, metadata_path: str, index_path: str = None):
        """
        初始化检索器，加载嵌入和元数据，构建或加载 FAISS 索引。

        Args:
            embeddings_path (str): 嵌入向量 .npy 文件路径。
            metadata_path (str): 元数据 .jsonl 文件路径。
            index_path (str, optional): FAISS 索引文件路径。如果提供，则加载现有索引。
        """
        logger.info(f"正在加载嵌入向量: {embeddings_path}")
        self.embeddings = np.load(embeddings_path).astype('float32') # FAISS 通常使用 float32
        logger.info(f"嵌入矩阵形状: {self.embeddings.shape}")

        logger.info(f"正在加载元数据: {metadata_path}")
        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.metadata.append(json.loads(line))
        logger.info(f"加载了 {len(self.metadata)} 条元数据。")

        if len(self.embeddings) != len(self.metadata):
            raise ValueError("嵌入向量的数量与元数据的数量不匹配！")

        if index_path and os.path.exists(index_path):
            logger.info(f"正在加载现有 FAISS 索引: {index_path}")
            self.index = faiss.read_index(index_path)
        else:
            logger.info("正在构建 FAISS 索引...")
            dimension = self.embeddings.shape[1] # 向量维度
            # 使用 IndexFlatIP (Inner Product) 作为简单的余弦相似度索引
            # 对于更大数据集，可以考虑 IndexLSH, IndexIVF, IndexHNSW 等
            self.index = faiss.IndexFlatIP(dimension)
            # 注意：对于余弦相似度，需要先对向量进行 L2 归一化
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            logger.info(f"FAISS 索引构建完成，包含 {self.index.ntotal} 个向量。")

            if index_path:
                logger.info(f"正在保存 FAISS 索引到: {index_path}")
                faiss.write_index(self.index, index_path)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[dict, float]]:
        """
        根据查询向量检索最相似的 k 个块。

        Args:
            query_embedding (np.ndarray): 查询向量。
            k (int): 检索结果数量。

        Returns:
            List[Tuple[dict, float]]: 包含 (metadata, similarity_score) 对的列表。
        """
        # 查询向量也需要 L2 归一化以计算余弦相似度
        query_embedding_norm = query_embedding.astype('float32').copy()
        faiss.normalize_L2(query_embedding_norm.reshape(1, -1))

        # 检索
        scores, indices = self.index.search(query_embedding_norm.reshape(1, -1), k)
        scores = scores[0] # 取第一个查询的结果
        indices = indices[0]

        # 根据索引获取元数据和分数
        results = []
        for idx, score in zip(indices, scores):
            if idx != -1 and idx < len(self.metadata): # 检查索引有效性
                results.append((self.metadata[idx], float(score))) # 转换 score 为 Python float
            else:
                logger.warning(f"检索到无效索引: {idx}")

        return results

if __name__ == "__main__":
    # --- 配置路径 ---
    EMBEDDINGS_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\embeddings.npy"
    METADATA_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\chunk_metadata.jsonl"
    # (可选) 保存索引文件，避免每次运行都重建
    INDEX_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\faiss_index.bin"

    # --- 初始化检索器 ---
    # 如果 INDEX_PATH 存在且你想加载它，传入 INDEX_PATH
    # retriever = VectorRetriever(EMBEDDINGS_PATH, METADATA_PATH, index_path=INDEX_PATH)
    # 如果你想重新构建索引并保存，传入 INDEX_PATH
    retriever = VectorRetriever(EMBEDDINGS_PATH, METADATA_PATH, index_path=INDEX_PATH)

    # --- 测试检索 ---
    # 这里需要一个嵌入模型来将测试查询转换为向量
    # 假设我们有一个 embedder 实例
    # from embedder import ChunkEmbedder
    # embedder = ChunkEmbedder()
    # query = "如何制作枫糖浆？"
    # query_embedding = embedder.embed_single_text(query)
    # results = retriever.search(query_embedding, k=3)
    # for i, (meta, score) in enumerate(results):
    #     print(f"--- Top {i+1} (Score: {score:.4f}) ---")
    #     print(f"Title: {meta['title']}")
    #     print(f"Text: {meta['text'][:200]}...") # 打印前200个字符
    #     print("-" * 80)

    logger.info("检索器初始化完成。")
