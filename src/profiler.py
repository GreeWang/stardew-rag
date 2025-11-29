# src/profiler.py

import time
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from lm_rag_system import RAGSystem
import os
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_latency_profiling(
    rag_system: RAGSystem,
    test_queries: List[str],
    num_runs_per_query: int = 3, # Run each query multiple times to get average
    config_label: str = "default"
) -> Dict[str, List[float]]:
    """
    对 RAG 系统进行延迟分析。

    Args:
        rag_system (RAGSystem): 已初始化的 RAG 系统。
        test_queries (List[str]): 用于测试的查询列表。
        num_runs_per_query (int): 每个查询运行的次数。
        config_label (str): 当前配置的标签。

    Returns:
        Dict[str, List[float]]: 包含各种延迟指标的字典。
    """
    logger.info(f"开始对配置 '{config_label}' 进行延迟分析...")

    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    for query in test_queries:
        for _ in range(num_runs_per_query):
            # Clear cache if enabled to get "cold" start time for profiling
            if rag_system.enable_caching:
                rag_system.cache.clear()
            # Run query and collect details
            _, details = rag_system.query(query, return_details=True)

            retrieval_latencies.append(details['retrieval_latency_s'])
            generation_latencies.append(details['generation_latency_s'])
            total_latencies.append(details['total_latency_s'])

    logger.info(f"配置 '{config_label}' 延迟分析完成。")

    return {
        'config_label': config_label,
        'retrieval_latencies': retrieval_latencies,
        'generation_latencies': generation_latencies,
        'total_latencies': total_latencies,
        'num_samples': len(total_latencies)
    }

def calculate_throughput(latencies: List[float]) -> float:
    """Calculate throughput based on latencies (requests per second)."""
    if not latencies:
        return 0.0
    avg_latency = np.mean(latencies)
    if avg_latency == 0:
        return float('inf') # Avoid division by zero
    return 1.0 / avg_latency

def plot_throughput_vs_quality(
    profile_results: List[Dict],
    quality_scores: List[float], # e.g., F1 scores from evaluation for each config
    output_path: str = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\plots"
):
    """
    Plot Throughput vs. Quality chart.

    Args:
        profile_results (List[Dict]): List of results from run_latency_profiling.
        quality_scores (List[float]): Quality scores corresponding to each config.
        output_path (str): Path to save the plot.
    """
    if len(profile_results) != len(quality_scores):
        raise ValueError("Number of profile results must match number of quality scores.")

    configs = [res['config_label'] for res in profile_results]
    throughputs = [calculate_throughput(res['total_latencies']) for res in profile_results]

    plt.figure(figsize=(10, 6))
    plt.scatter(quality_scores, throughputs, s=100, alpha=0.7)
    plt.xlabel('Quality Score (e.g., F1)')
    plt.ylabel('Throughput (Queries/Second)')
    plt.title('Throughput vs. Quality for Different RAG Configurations')
    # Annotate points with config labels
    for i, txt in enumerate(configs):
        plt.annotate(txt, (quality_scores[i], throughputs[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Throughput vs. Quality 图已保存至: {output_path}")
    plt.show() # Or plt.close() if running in a non-interactive environment

if __name__ == "__main__":
    # --- Configuration ---
    EMBEDDINGS_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\embeddings.npy"
    METADATA_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\chunk_metadata.jsonl"
    INDEX_PATH = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data\\faiss_index.bin"
    EMBEDDING_MODEL_NAME = "moka-ai/m3e-base"
    LLM_MODEL_NAME = "gpt-3.5-turbo" # Or "llama3"
    API_KEY = "sk-tT5HcopxjJ7vGdnX4333Ef20D1E44eB7827b98D4A923F9E2" # Or None if using Ollama

    # Define configurations to test
    configs_to_test = [
        {"max_tokens_for_generation": 256, "top_k_retrieval": 3, "enable_caching": False, "label": "NoCache_256_3"},
        {"max_tokens_for_generation": 256, "top_k_retrieval": 3, "enable_caching": True, "label": "Cache_256_3"},
        {"max_tokens_for_generation": 512, "top_k_retrieval": 5, "enable_caching": False, "label": "NoCache_512_5"},
        {"max_tokens_for_generation": 512, "top_k_retrieval": 5, "enable_caching": True, "label": "Cache_512_5"},
    ]

    # Example test queries (use a subset of your evaluation set or create new ones)
    test_queries = [
        "星露谷物语中如何制作枫糖浆？",
        "矿车怎么用？",
        "腌鱼籽怎么做？",
        "如何获得铱锭？",
        "沙漠区域有什么？"
    ]

    # Example quality scores for each config (these would come from your evaluation script)
    # Replace these with actual scores from your evaluation!
    example_quality_scores = [0.75, 0.74, 0.80, 0.79] # Placeholder values

    profile_results = []

    for config in configs_to_test:
        rag_system = RAGSystem(
            embeddings_path=EMBEDDINGS_PATH,
            metadata_path=METADATA_PATH,
            index_path=INDEX_PATH,
            model_name=EMBEDDING_MODEL_NAME,
            llm_model_name=LLM_MODEL_NAME,
            api_key=API_KEY,
            max_tokens_for_generation=config['max_tokens_for_generation'],
            top_k_retrieval=config['top_k_retrieval'],
            enable_caching=config['enable_caching']
        )

        result = run_latency_profiling(
            rag_system=rag_system,
            test_queries=test_queries,
            num_runs_per_query=3, # Adjust as needed
            config_label=config['label']
        )
        profile_results.append(result)

    # Print summary
    for res in profile_results:
        avg_total_lat = np.mean(res['total_latencies'])
        avg_retrieval_lat = np.mean(res['retrieval_latencies'])
        avg_generation_lat = np.mean(res['generation_latencies'])
        throughput = calculate_throughput(res['total_latencies'])
        print(f"\n配置: {res['config_label']}")
        print(f"  平均总延迟: {avg_total_lat:.4f}s")
        print(f"  平均检索延迟: {avg_retrieval_lat:.4f}s")
        print(f"  平均生成延迟: {avg_generation_lat:.4f}s")
        print(f"  吞吐量: {throughput:.2f} QPS")
        print(f"  样本数: {res['num_samples']}")

    # Plot Throughput vs Quality
    # Ensure you have the correct quality scores for your configs!
    plot_throughput_vs_quality(profile_results, example_quality_scores)

    # Optional: Save raw latency data
    # raw_data_path = "../data/profile_raw_data.json"
    # with open(raw_data_path, 'w', encoding='utf-8') as f:
    #     json.dump(profile_results, f, ensure_ascii=False, indent=2)
    # logger.info(f"原始延迟数据已保存至: {raw_data_path}")
