# sparse_retriever.py
from rank_bm25 import BM25Okapi
import jieba
import json
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self, metadata_path: str):
        self.metadata = self.load_metadata(metadata_path)
        if not self.metadata:
            raise ValueError(f"无法从 {metadata_path} 加载元数据")
        
        self.corpus = [self.tokenize(item["text"]) for item in self.metadata]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"BM25检索器初始化完成，共加载 {len(self.metadata)} 个文档")
    
    def load_metadata(self, metadata_path: str) -> List[dict]:
        """加载元数据，与VectorRetriever相同"""
        logger.info(f"正在加载元数据: {metadata_path}")
        try:
            metadata = []
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        metadata.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析JSON行失败: {e}，跳过该行")
                        continue
            
            logger.info(f"成功加载 {len(metadata)} 条元数据")
            return metadata
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return []
    
    def tokenize(self, text: str) -> List[str]:
        """使用jieba进行中文分词"""
        if not text:
            return []
        return list(jieba.cut(text))
    
    def search(self, query: str, k: int = 5) -> List[Tuple[dict, float]]:
        """执行BM25检索"""
        if not query:
            return []
        
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((self.metadata[idx], float(scores[idx])))
        
        return results