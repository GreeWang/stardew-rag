# rag_evaluator.py
import json
import os
import re
import numpy as np
from typing import List, Dict
import logging
from datetime import datetime
from pathlib import Path
from config import load_config, DATA_DIR, EVAL_RESULTS_DIR

from evaluation_set_creator import EvaluationSetCreator
from sparse_retriever import BM25Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, base_path: str | Path = DATA_DIR, force_create_eval_set: bool = False):
        self.base_path = str(base_path)
        self.evaluation_set = []
        self.evaluation_set_creator = EvaluationSetCreator(self.base_path)

        if force_create_eval_set or not self.evaluation_set_exists():
            self.create_evaluation_set()
        else:
            self.load_evaluation_set()

    def create_evaluation_set(self, num_items: int = 60) -> List[Dict]:
        """从原始文档生成评估集。"""
        self.evaluation_set = self.evaluation_set_creator.create_evaluation_set(num_items)
        return self.evaluation_set

    def evaluation_set_exists(self) -> bool:
        """检查评估集是否存在"""
        eval_path = os.path.join(self.base_path, "evaluation_set.json")
        return os.path.exists(eval_path)
    
    
    def load_evaluation_set(self) -> List[Dict]:
        """加载评估集"""
        eval_path = os.path.join(self.base_path, "evaluation_set.json")
        if os.path.exists(eval_path):
            try:
                with open(eval_path, 'r', encoding='utf-8') as f:
                    self.evaluation_set = json.load(f)
                logger.info(f"成功加载包含 {len(self.evaluation_set)} 个问题的评估集")
                return self.evaluation_set
            except Exception as e:
                logger.error(f"加载评估集失败: {e}")
        else:
            logger.error("评估集文件不存在")
        return []
    
    def clean_wiki_text(self, text: str) -> str:
        """
        清理wiki文本，只移除"来自Stardew Valley Wiki"固定句子
        """
        if not text:
            return ""
        
        cleaned_text = text
        
        # 只移除"来自Stardew Valley Wiki"及其变体
        patterns_to_remove = [
            r"来自Stardew Valley Wiki",
            r"来自.*?Stardew Valley Wiki",
        ]
        
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, "", cleaned_text)
        
        # 清理多余的空格（但保留单个空格）
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def evaluate_retrieval(self, rag_system, k_values: List[int] = [1, 3, 5]) -> Dict:
        """
        评估检索性能
        """
        if not self.evaluation_set:
            logger.error("评估集为空，无法进行评估")
            return {}
        
        results = {}
        reciprocal_ranks = []  # 用于计算MRR
        
        for k in k_values:
            recall_at_k = []
            
            for item in self.evaluation_set:
                question = item["question"]
                true_sources = item["true_sources"]
                
                # 获取检索结果
                try:
                    q_emb = rag_system.embedder.embed_single_text(question)
                    # 如果是通过 ComparisonExperiment 创建的 CustomRAGSystem，
                    # 它会有 retriever_type='sparse'；另外也检测 BM25Retriever 实例。
                    is_sparse = getattr(rag_system, 'retriever_type', None) == 'sparse' or isinstance(rag_system.retriever, BM25Retriever)
                    if is_sparse:
                        retrieved = rag_system.retriever.search(question, k=k)
                    else:
                        # 确保传入的是 numpy 数组（防止 list/tuple 导致歧义）
                        if isinstance(q_emb, (list, tuple)):
                            q_emb = np.array(q_emb, dtype='float32')
                        retrieved = rag_system.retriever.search(q_emb, k=k)
                    
                    # 计算召回率
                    retrieved_titles = [meta["title"] for meta, score in retrieved]
                    
                    # 清理标题以便更好地匹配
                    clean_true_sources = [source.split(" - ")[0] if " - " in source else source for source in true_sources]
                    clean_retrieved_titles = [title.split(" - ")[0] if " - " in title else title for title in retrieved_titles]
                    
                    matched_sources = set(clean_retrieved_titles) & set(clean_true_sources)
                    recall = len(matched_sources) / len(clean_true_sources) if clean_true_sources else 0
                    recall_at_k.append(recall)
                    
                    # 计算倒数排名 (用于MRR)
                    if k == max(k_values):  # 只在最大k值时计算MRR
                        reciprocal_rank = 0
                        for rank, title in enumerate(clean_retrieved_titles, 1):
                            if title in clean_true_sources:
                                reciprocal_rank = 1.0 / rank
                                break
                        reciprocal_ranks.append(reciprocal_rank)
                    
                except Exception as e:
                    logger.error(f"评估问题 '{question}' 时出错: {e}")
                    recall_at_k.append(0)
                    if k == max(k_values):
                        reciprocal_ranks.append(0)
            
            results[f"recall@{k}"] = np.mean(recall_at_k)
            results[f"recall@{k}_std"] = np.std(recall_at_k)
        
        # 计算平均召回率
        results["mean_recall"] = np.mean([results[f"recall@{k}"] for k in k_values])
        
        # 计算MRR (平均倒数排名)
        results["mrr"] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        
        return results
    
    def calculate_f1_score(self, generated_answer: str, true_answer: str) -> float:
        """
        计算生成答案和真实答案之间的F1分数
        """
        # 使用更好的中文分词
        import jieba
        gen_words = set(jieba.cut(generated_answer))
        true_words = set(jieba.cut(true_answer))
        
        # 过滤停用词和短词
        stop_words = {"的", "是", "在", "有", "和", "与", "或", "了", "着", "过"}
        gen_words = {word for word in gen_words if len(word) > 1 and word not in stop_words}
        true_words = {word for word in true_words if len(word) > 1 and word not in stop_words}
        
        # 如果任一集合为空，返回0
        if not gen_words or not true_words:
            return 0.0
        
        # 计算交集
        common_words = gen_words & true_words
        
        # 计算精确率和召回率
        precision = len(common_words) / len(gen_words)
        recall = len(common_words) / len(true_words)
        
        # 计算F1分数
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_answer_quality(self, rag_system, sample_size: int = None) -> Dict:
        """
        评估答案质量
        """
        if not self.evaluation_set:
            logger.error("评估集为空，无法进行评估")
            return {}
        
        # 确定要评估的样本
        if sample_size and sample_size < len(self.evaluation_set):
            import random
            samples = random.sample(self.evaluation_set, sample_size)
        else:
            samples = self.evaluation_set
        
        results = {
            "answers": [],
            "faithfulness_scores": [],
            "relevance_scores": [],
            "completeness_scores": [],
            "f1_scores": []
        }
        
        for i, item in enumerate(samples):
            question = item["question"]
            true_answer = item["true_answer"]
            true_sources = item["true_sources"]
            
            logger.info(f"评估进度: {i+1}/{len(samples)} - {question}")

            try:
                # 获取检索结果
                q_emb = rag_system.embedder.embed_single_text(question)
                is_sparse = getattr(rag_system, 'retriever_type', None) == 'sparse' or isinstance(rag_system.retriever, BM25Retriever)
                if is_sparse:
                    retrieved = rag_system.retriever.search(question, k=5)
                else:
                    if isinstance(q_emb, (list, tuple)):
                        q_emb = np.array(q_emb, dtype='float32')
                    retrieved = rag_system.retriever.search(q_emb, k=5)
                chunks = [meta for meta, score in retrieved]

                # 生成答案
                generated_answer = rag_system.generator.generate_answer(question, chunks)

                # 计算相似度指标
                faithfulness = self.calculate_faithfulness(generated_answer, chunks)
                relevance = self.calculate_relevance(generated_answer, true_sources)
                completeness = self.calculate_completeness(generated_answer, true_answer)
                f1_score = self.calculate_f1_score(generated_answer, true_answer)

                results["answers"].append({
                    "question": question,
                    "true_answer": true_answer,
                    "generated_answer": generated_answer,
                    "true_sources": true_sources,
                    "retrieved_sources": [chunk["title"] for chunk in chunks]
                })

                results["faithfulness_scores"].append(faithfulness)
                results["relevance_scores"].append(relevance)
                results["completeness_scores"].append(completeness)
                results["f1_scores"].append(f1_score)

            except Exception as e:
                logger.error(f"评估问题 '{question}' 时出错: {e}")
                # 添加默认值
                results["faithfulness_scores"].append(0)
                results["relevance_scores"].append(0)
                results["completeness_scores"].append(0)
                results["f1_scores"].append(0)
        
        # 计算平均分
        results["avg_faithfulness"] = np.mean(results["faithfulness_scores"])
        results["avg_relevance"] = np.mean(results["relevance_scores"])
        results["avg_completeness"] = np.mean(results["completeness_scores"])
        results["avg_f1"] = np.mean(results["f1_scores"])
        results["overall_score"] = (results["avg_faithfulness"] + 
                                   results["avg_relevance"] + 
                                   results["avg_completeness"]) / 3
        
        return results
    
    def calculate_faithfulness(self, answer: str, chunks: List[Dict]) -> float:
        """计算答案忠实度（基于检索到的内容）"""
        chunk_titles = [chunk["title"] for chunk in chunks]
        # 清理标题
        clean_chunk_titles = [title.split(" - ")[0] if " - " in title else title for title in chunk_titles]
        title_mentions = sum(1 for title in clean_chunk_titles if title in answer)
        return title_mentions / len(clean_chunk_titles) if clean_chunk_titles else 0
    
    def calculate_relevance(self, answer: str, true_sources: List[str]) -> float:
        """计算答案相关性 - 基于true_sources中的关键词"""
        if not answer or not true_sources:
            return 0.0
        
        # 清理true_sources中的关键词（移除wiki后缀等）
        clean_keywords = []
        for source in true_sources:
            # 移除" - 星露谷物语官方中文维基"等后缀
            if " - " in source:
                keyword = source.split(" - ")[0]
            else:
                keyword = source
            clean_keywords.append(keyword)
        
        # 计算匹配的关键词数量
        matched_keywords = 0
        for keyword in clean_keywords:
            # 使用更宽松的匹配方式，允许部分匹配
            if keyword in answer:
                matched_keywords += 1
            else:
                # 如果完整匹配失败，尝试部分匹配（对于较长的关键词）
                if len(keyword) > 2:
                    # 尝试匹配关键词的前2/3部分
                    partial_keyword = keyword[:int(len(keyword)*2/3)]
                    if partial_keyword in answer:
                        matched_keywords += 0.7  # 部分匹配给部分分数
                    else:
                        # 尝试匹配关键词的前1/2部分
                        partial_keyword = keyword[:int(len(keyword)*1/2)]
                        if partial_keyword in answer:
                            matched_keywords += 0.5
        
        # 计算相关性分数
        if not clean_keywords:
            return 0.0
        
        relevance_score = matched_keywords / len(clean_keywords)
        return min(relevance_score, 1.0)  # 确保不超过1.0


    
    def calculate_completeness(self, generated_answer: str, true_answer: str) -> float:
        """计算答案完整性"""
        # 清理生成答案和真实答案
        cleaned_generated = self.clean_wiki_text(generated_answer)
        cleaned_true = self.clean_wiki_text(true_answer)
        
        gen_len = len(cleaned_generated)
        true_len = len(cleaned_true)
        
        if true_len == 0:
            return 0
        
        # 限制在0-1之间
        ratio = min(gen_len / true_len, 1.0)
        return ratio
    
    def run_complete_evaluation(self, rag_system, output_dir: str = str(EVAL_RESULTS_DIR)):
        """
        运行完整的评估流程
        """
        logger.info("开始完整RAG系统评估")
        
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. 评估检索性能
        logger.info("评估检索性能...")
        retrieval_results = self.evaluate_retrieval(rag_system)
        
        # 2. 评估答案质量
        logger.info("评估答案质量...")
        quality_results = self.evaluate_answer_quality(rag_system, sample_size=30)
        
        # 整合所有结果
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_set_size": len(self.evaluation_set),
            "evaluation_set_stats": self.evaluation_set_creator.get_evaluation_set_stats(),
            "retrieval_metrics": retrieval_results,
            "answer_quality": quality_results
        }
        
        # 保存结果
        results_file = os.path.join(output_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        # 生成摘要报告
        self.generate_summary_report(final_results, output_dir)
        
        logger.info(f"评估完成！结果已保存至 {results_file}")
        
        return final_results
    
    def generate_summary_report(self, results: Dict, output_dir: str):
        """生成评估摘要报告"""
        retrieval = results["retrieval_metrics"]
        quality = results["answer_quality"]
        stats = results["evaluation_set_stats"]
        
        report = f"""
RAG系统评估报告
================

生成时间: {results['timestamp']}
评估集大小: {results['evaluation_set_size']}

评估集统计
----------
- 总问题数: {stats.get('total_questions', 0)}
- 难度分布: {stats.get('difficulty_distribution', {})}

检索性能指标
------------
- 平均召回率: {retrieval.get('mean_recall', 0):.4f}
- Recall@1: {retrieval.get('recall@1', 0):.4f}
- Recall@3: {retrieval.get('recall@3', 0):.4f}
- Recall@5: {retrieval.get('recall@5', 0):.4f}
- MRR (平均倒数排名): {retrieval.get('mrr', 0):.4f}

答案质量指标
------------
- 平均忠实度: {quality.get('avg_faithfulness', 0):.4f}
- 平均相关性: {quality.get('avg_relevance', 0):.4f}
- 平均完整性: {quality.get('avg_completeness', 0):.4f}
- 平均F1分数: {quality.get('avg_f1', 0):.4f}
- 整体质量分数: {quality.get('overall_score', 0):.4f}

评估详情
--------
- 检索性能评估了系统找到相关文档的能力
- 答案质量评估了生成答案的准确性、相关性和完整性
- 忠实度: 答案是否基于检索到的内容
- 相关性: 答案是否与问题相关
- 完整性: 答案是否提供了足够的信息
- F1分数: 生成答案与真实答案的词汇重叠程度
- MRR: 第一个相关文档的平均排名质量
"""
        
        report_file = os.path.join(output_dir, "evaluation_summary.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 在控制台打印简洁的报告
        print("\n" + "="*50)
        print("RAG系统评估结果")
        print("="*50)
        print(f"检索性能:")
        print(f"  - 平均召回率: {retrieval.get('mean_recall', 0):.4f}")
        print(f"  - Recall@1: {retrieval.get('recall@1', 0):.4f}")
        print(f"  - Recall@3: {retrieval.get('recall@3', 0):.4f}")
        print(f"  - Recall@5: {retrieval.get('recall@5', 0):.4f}")
        print(f"  - MRR: {retrieval.get('mrr', 0):.4f}")
        print(f"答案质量:")
        print(f"  - 整体分数: {quality.get('overall_score', 0):.4f}")
        print(f"  - 忠实度: {quality.get('avg_faithfulness', 0):.4f}")
        print(f"  - 相关性: {quality.get('avg_relevance', 0):.4f}")
        print(f"  - 完整性: {quality.get('avg_completeness', 0):.4f}")
        print(f"  - F1分数: {quality.get('avg_f1', 0):.4f}")
        print("="*50)


def main():
    """主函数 - 运行评估"""
    paths, models, pipeline = load_config()

    # 初始化评估器
    evaluator = RAGEvaluator(base_path=paths.base_dir if hasattr(paths, "base_dir") else paths.raw_docs.parent, force_create_eval_set=False)
    
    print(f"已加载包含 {len(evaluator.evaluation_set)} 个问题的评估集")
    
    # 初始化RAG系统
    print("初始化RAG系统...")
    from rag_system import RAGSystem
    
    try:
        rag_system = RAGSystem(
            embeddings_path=str(paths.embeddings),
            metadata_path=str(paths.chunk_metadata),
            index_path=str(paths.faiss_index),
            model_name=models.embed_model,
            llm_model_name=models.llm_model,
            api_key=models.openai_api_key,
            base_url=models.openai_base_url
        )
    except Exception as e:
        print(f"初始化RAG系统失败: {e}")
        print("请确保已运行 rag_system.py 完成数据处理")
        return
    
    # 运行完整评估
    print("运行完整评估...")
    results = evaluator.run_complete_evaluation(rag_system)
    
    print("评估完成！")


if __name__ == "__main__":
    main()
