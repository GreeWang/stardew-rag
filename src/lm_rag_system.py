# src/rag_system.py

import logging
import time
import psutil # Add psutil import
import os
from typing import List, Dict, Tuple
from embedder import ChunkEmbedder
from retriever import VectorRetriever
from lm_generator import RAGGenerator # Or LocalRAGGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        index_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        # --- Add profiling parameters ---
        max_tokens_for_generation: int = 512, # Allow changing max_tokens
        top_k_retrieval: int = 5,            # Allow changing top_k
        enable_caching: bool = False         # Allow enabling/disabling caching
    ):
        """
        初始化完整的 RAG 系统。

        Args:
            # ... (previous args) ...
            max_tokens_for_generation (int): LLM 生成答案时的最大 token 长度。
            top_k_retrieval (int): 检索相关块的数量。
            enable_caching (bool): 是否启用缓存。
        """
        logger.info("正在初始化 RAG 系统...")

        self.embedder = ChunkEmbedder(model_name=model_name)
        self.retriever = VectorRetriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            index_path=index_path
        )
        self.generator = RAGGenerator(api_key=api_key, model_name=llm_model_name)
        # If using Ollama: self.generator = LocalRAGGenerator(model_name=llm_model_name)

        # --- Profiling parameters ---
        self.max_tokens_for_generation = max_tokens_for_generation
        self.top_k_retrieval = top_k_retrieval
        self.enable_caching = enable_caching

        # --- Caching setup (basic example using a dictionary) ---
        if self.enable_caching:
            self.cache = {}
            logger.info("缓存已启用。")
        else:
            self.cache = None
            logger.info("缓存已禁用。")

        logger.info("RAG 系统初始化完成。")

    def query(self, user_question: str, return_details: bool = False) -> str | Tuple[str, Dict]:
        """
        处理用户查询，返回 RAG 生成的答案。

        Args:
            user_question (str): 用户的问题。
            return_details (bool): 是否返回包含延迟等详细信息的元组。

        Returns:
            str: 最终答案。
            或 Tuple[str, Dict]: (答案, 详细信息字典)
        """
        # --- Caching Check ---
        if self.enable_caching and user_question in self.cache:
            cached_answer, cached_details = self.cache[user_question]
            logger.info(f"缓存命中: '{user_question[:30]}...'")
            if return_details:
                # Update details to indicate cache hit
                cached_details['cached'] = True
                return cached_answer, cached_details
            else:
                return cached_answer

        logger.info(f"收到查询: '{user_question}'")

        # --- Measure Retrieval Latency ---
        start_time = time.perf_counter()
        query_embedding = self.embedder.embed_single_text(user_question)
        embed_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        retrieved_results = self.retriever.search(query_embedding, k=self.top_k_retrieval)
        retrieval_time = time.perf_counter() - start_time
        retrieved_chunks = [meta for meta, score in retrieved_results]

        if not retrieved_chunks:
            logger.warning("未检索到相关块。")
            answer = "根据提供的信息，我无法回答这个问题。"
            details = {
                'retrieval_latency_s': retrieval_time + embed_time,
                'generation_latency_s': 0.0,
                'total_latency_s': retrieval_time + embed_time,
                'cached': False,
                'retrieved_chunks_count': 0,
                # 'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 # Memory usage might be more stable measured externally
            }
            if return_details:
                return answer, details
            else:
                return answer

        # --- Measure Generation Latency ---
        start_time = time.perf_counter()
        # If using OpenAI generator, pass max_tokens
        # You need to modify generator.generate_answer to accept max_tokens
        answer = self.generator.generate_answer(user_question, retrieved_chunks, max_tokens=self.max_tokens_for_generation)
        generation_time = time.perf_counter() - start_time

        total_time = embed_time + retrieval_time + generation_time
        details = {
            'retrieval_latency_s': retrieval_time + embed_time,
            'generation_latency_s': generation_time,
            'total_latency_s': total_time,
            'cached': False,
            'retrieved_chunks_count': len(retrieved_chunks),
            # 'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 # Memory usage might be more stable measured externally
        }

        # --- Caching Store ---
        if self.enable_caching:
            self.cache[user_question] = (answer, details.copy()) # Store a copy of details

        logger.info(f"查询处理完成，总延迟: {total_time:.4f}s")

        if return_details:
            return answer, details
        else:
            return answer

    def get_memory_usage_mb(self):
        """Get current process memory usage."""
        return psutil.Process().memory_info().rss / 1024 / 1024


if __name__ == "__main__":
    # --- 配置路径和模型 ---
    EMBEDDINGS_PATH = "../data/embeddings.npy"
    METADATA_PATH = "../data/chunk_metadata.jsonl"
    INDEX_PATH = "../data/faiss_index.bin"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    LLM_MODEL_NAME = "gpt-3.5-turbo" # Or "llama3"
    API_KEY = os.getenv("OPENAI_API_KEY") # Or None if using Ollama

    # --- Test different configurations ---
    configs_to_test = [
        {"max_tokens_for_generation": 256, "top_k_retrieval": 3, "enable_caching": False},
        {"max_tokens_for_generation": 256, "top_k_retrieval": 3, "enable_caching": True},
        {"max_tokens_for_generation": 512, "top_k_retrieval": 5, "enable_caching": False},
        {"max_tokens_for_generation": 512, "top_k_retrieval": 5, "enable_caching": True},
    ]

    test_query = "星露谷物语中如何制作枫糖浆？"

    for i, config in enumerate(configs_to_test):
        print(f"\n--- 测试配置 {i+1}: {config} ---")
        rag_system = RAGSystem(
            embeddings_path=EMBEDDINGS_PATH,
            metadata_path=METADATA_PATH,
            index_path=INDEX_PATH,
            model_name=EMBEDDING_MODEL_NAME,
            llm_model_name=LLM_MODEL_NAME,
            api_key=API_KEY,
            **config # Pass config parameters
        )

        # Warm up (optional, to reduce variance)
        # rag_system.query(test_query)

        # Measure latency for a single query
        answer, details = rag_system.query(test_query, return_details=True)
        print(f"  答案: {answer[:100]}...") # Print first 100 chars
        print(f"  检索延迟: {details['retrieval_latency_s']:.4f}s")
        print(f"  生成延迟: {details['generation_latency_s']:.4f}s")
        print(f"  总延迟: {details['total_latency_s']:.4f}s")
        print(f"  缓存命中: {details['cached']}")
        print(f"  检索块数: {details['retrieved_chunks_count']}")
        # print(f"  内存使用: {details['memory_usage_mb']:.2f} MB") # May not be stable within function

        # Test caching by querying again
        if config['enable_caching']:
            print("  (再次查询相同问题以测试缓存)...")
            _, details_cached = rag_system.query(test_query, return_details=True)
            print(f"  缓存命中延迟: {details_cached['total_latency_s']:.4f}s")
