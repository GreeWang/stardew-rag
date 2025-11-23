# comparison_experiment.py
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging
import shutil

# 导入必要的模块
from embedder import ChunkEmbedder
from retriever import VectorRetriever
from generator import RAGGenerator
from evaluation_set_creator import EvaluationSetCreator
from rag_evaluator import RAGEvaluator
from sparse_retriever import BM25Retriever  # 新增稀疏检索器

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComparisonExperiment:
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        self.results = {}
        
    def ensure_directory_exists(self, path: str):
        """确保目录存在，如果不存在则创建"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
    def generate_embeddings_for_model(self, model_name: str):
        """为指定模型生成嵌入文件"""
        # 生成模型特定的文件名
        model_safe_name = model_name.replace('/', '_')
        embeddings_path = f"{self.base_path}\\embeddings_{model_safe_name}.npy"
        metadata_path = f"{self.base_path}\\chunk_metadata_{model_safe_name}.jsonl"
        index_path = f"{self.base_path}\\faiss_index_{model_safe_name}.bin"
        
        # 检查是否已存在嵌入文件
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            logger.info(f"嵌入文件已存在: {embeddings_path}")
            return embeddings_path, metadata_path, index_path
        
        logger.info(f"为模型 {model_name} 生成嵌入文件...")
        
        # 使用指定的模型生成嵌入
        embedder = ChunkEmbedder(model_name=model_name)
        
        # 生成嵌入
        embedder.embed_chunks(
            chunks_file_path=f"{self.base_path}\\chunks.jsonl",
            output_embeddings_path=embeddings_path,
            output_metadata_path=metadata_path
        )
        
        logger.info(f"嵌入文件生成完成: {embeddings_path}")
        return embeddings_path, metadata_path, index_path
    
    def initialize_sparse_retriever(self):
        """初始化稀疏检索器 (BM25)"""
        logger.info("初始化稀疏检索器 (BM25)...")
        
        # 使用m3e模型的元数据文件，确保文件存在
        metadata_path = f"{self.base_path}\\chunk_metadata_moka-ai_m3e-base.jsonl"
        
        # 如果文件不存在，尝试使用默认的元数据文件
        if not os.path.exists(metadata_path):
            logger.warning(f"元数据文件 {metadata_path} 不存在，尝试使用默认元数据文件")
            metadata_path = f"{self.base_path}\\chunk_metadata.jsonl"
            
            # 如果默认文件也不存在，则生成一个
            if not os.path.exists(metadata_path):
                logger.error("找不到任何元数据文件，无法初始化稀疏检索器")
                raise FileNotFoundError("找不到元数据文件")
        
        logger.info(f"使用元数据文件: {metadata_path}")
        
        # 初始化BM25检索器
        retriever = BM25Retriever(metadata_path=metadata_path)
        
        logger.info("稀疏检索器初始化完成")
        return retriever
    
    def initialize_rag_system(self, config: Dict):
        """根据配置初始化RAG系统"""
        logger.info(f"初始化RAG系统: {config['name']}")
        
        # 根据检索器类型初始化不同的组件
        if config["retriever_type"] == "sparse":
            # 初始化稀疏检索器
            retriever = self.initialize_sparse_retriever()
            
            # 对于稀疏检索器，我们不需要嵌入器，但为了保持接口一致，创建一个假的嵌入器
            class DummyEmbedder:
                def embed_single_text(self, text: str):
                    # 返回一个空的嵌入向量（稀疏检索器不使用嵌入）
                    return np.zeros(384)  # 假设维度为384
                    
            embedder = DummyEmbedder()
            
        else:  # 密集检索器
            # 为指定模型生成嵌入文件
            embeddings_path, metadata_path, index_path = self.generate_embeddings_for_model(config["embed_model"])
            
            # 初始化嵌入器
            embedder = ChunkEmbedder(model_name=config["embed_model"])
            
            # 初始化检索器
            retriever = VectorRetriever(
                embeddings_path=embeddings_path,
                metadata_path=metadata_path,
                index_path=index_path
            )
        
        # 初始化生成器
        generator = RAGGenerator(
            api_key="sk-tT5HcopxjJ7vGdnX4333Ef20D1E44eB7827b98D4A923F9E2",
            model_name="gpt-4-turbo",
            base_url="https://bj.yi-zhan.top/v1",
            prompt_type=config["prompt_type"]
        )
        
        # 创建自定义RAG系统
        class CustomRAGSystem:
            def __init__(self, embedder, retriever, generator, retriever_type):
                self.embedder = embedder
                self.retriever = retriever
                self.generator = generator
                self.retriever_type = retriever_type
            
            def query(self, question: str, top_k=5):
                if self.retriever_type == "sparse":
                    # 稀疏检索器直接使用文本查询
                    retrieved = self.retriever.search(question, k=top_k)
                else:
                    # 密集检索器需要先嵌入查询
                    q_emb = self.embedder.embed_single_text(question)
                    retrieved = self.retriever.search(q_emb, k=top_k)
                    
                chunks = [meta for meta, score in retrieved]
                answer = self.generator.generate_answer(question, chunks)
                return answer
        
        return CustomRAGSystem(embedder, retriever, generator, config["retriever_type"])
    
    def run_comparison_experiments(self):
        """运行比较实验"""
        # 实验配置 - 添加了稀疏检索器实验
        EXPERIMENT_CONFIGS = [
            # 密集检索器实验
            {
                "name": "m3e_base_standard",
                "embed_model": "moka-ai/m3e-base",
                "retriever_type": "dense",
                "prompt_type": "standard"
            },
            {
                "name": "m3e_base_detailed",
                "embed_model": "moka-ai/m3e-base", 
                "retriever_type": "dense",
                "prompt_type": "detailed"
            },
            {
                "name": "m3e_base_stardew",
                "embed_model": "moka-ai/m3e-base",
                "retriever_type": "dense", 
                "prompt_type": "stardew_specific"
            },
            {
                "name": "bge_small_standard",
                "embed_model": "BAAI/bge-small-zh",
                "retriever_type": "dense",
                "prompt_type": "standard"
            },
            # 稀疏检索器实验
            {
                "name": "bm25_standard",
                "embed_model": "none",  # 稀疏检索器不需要嵌入模型
                "retriever_type": "sparse",
                "prompt_type": "standard"
            },
            {
                "name": "bm25_detailed",
                "embed_model": "none",
                "retriever_type": "sparse",
                "prompt_type": "detailed"
            }
        ]
        
        # 确保评估集存在
        evaluator = RAGEvaluator(force_create_eval_set=False)
        if not evaluator.evaluation_set:
            logger.error("评估集不存在，请先运行 evaluation_set_creator.py")
            return
        
        # 运行每个实验
        for config in EXPERIMENT_CONFIGS:
            logger.info(f"开始实验: {config['name']}")
            
            try:
                # 确保输出目录存在
                output_dir = f"comparison_results/{config['name']}"
                self.ensure_directory_exists(output_dir)
                
                # 初始化RAG系统
                rag_system = self.initialize_rag_system(config)
                
                # 运行评估
                experiment_evaluator = RAGEvaluator(force_create_eval_set=False)
                experiment_results = experiment_evaluator.run_complete_evaluation(
                    rag_system, 
                    output_dir=output_dir
                )
                
                self.results[config['name']] = {
                    "config": config,
                    "results": experiment_results
                }
                
                logger.info(f"实验 {config['name']} 完成")
                
            except Exception as e:
                logger.error(f"实验 {config['name']} 失败: {e}")
                self.results[config['name']] = {
                    "config": config,
                    "error": str(e)
                }
        
        # 保存所有实验结果
        self.save_all_results()
        
        # 生成比较报告
        self.generate_comparison_report()
        
        return self.results
    
    def save_all_results(self):
        """保存所有实验结果"""
        output_dir = "comparison_results"
        self.ensure_directory_exists(output_dir)
        
        results_file = os.path.join(output_dir, f"all_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"所有实验结果已保存至: {results_file}")
    
    def generate_comparison_report(self):
        """生成比较实验报告"""
        if not self.results:
            logger.error("没有实验结果可分析")
            return
        
        # 生成比较表格
        comparison_table = self._generate_comparison_table()
        
        # 生成详细报告
        detailed_report = self._generate_detailed_report()
        
        # 保存报告
        output_dir = "comparison_results"
        self.ensure_directory_exists(output_dir)
        
        report_file = os.path.join(output_dir, "comparison_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RAG系统比较实验报告\n\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
            f.write(comparison_table)
            f.write("\n\n")
            f.write(detailed_report)
        
        # 在控制台打印简洁的比较结果
        print("\n" + "="*80)
        print("RAG系统比较实验结果")
        print("="*80)
        print(comparison_table)
        print("="*80)
        
        logger.info(f"比较报告已保存至: {report_file}")
    
    def _generate_comparison_table(self):
        """生成比较表格"""
        table = "## 检索性能比较\n\n"
        table += "| 实验配置 | 检索器类型 | Recall@1 | Recall@3 | Recall@5 | MRR |\n"
        table += "|----------|------------|----------|----------|----------|-----|\n"
        
        for exp_name, data in self.results.items():
            if "error" in data:
                table += f"| {exp_name} | {data['config']['retriever_type']} | 错误 | 错误 | 错误 | 错误 |\n"
                continue
            
            retrieval = data["results"]["retrieval_metrics"]
            table += f"| {exp_name} | {data['config']['retriever_type']} | {retrieval['recall@1']:.4f} | {retrieval['recall@3']:.4f} | {retrieval['recall@5']:.4f} | {retrieval['mrr']:.4f} |\n"
        
        table += "\n## 答案质量比较\n\n"
        table += "| 实验配置 | 检索器类型 | 忠实度 | 相关性 | 完整性 | F1 | 整体质量 |\n"
        table += "|----------|------------|---------|---------|---------|----|----------|\n"
        
        for exp_name, data in self.results.items():
            if "error" in data:
                table += f"| {exp_name} | {data['config']['retriever_type']} | 错误 | 错误 | 错误 | 错误 | 错误 |\n"
                continue
            
            quality = data["results"]["answer_quality"]
            table += f"| {exp_name} | {data['config']['retriever_type']} | {quality['avg_faithfulness']:.4f} | {quality['avg_relevance']:.4f} | {quality['avg_completeness']:.4f} | {quality['avg_f1']:.4f} | {quality['overall_score']:.4f} |\n"
        
        return table
    
    def _generate_detailed_report(self):
        """生成详细报告"""
        report = "## 详细分析\n\n"
        
        # 找出最佳配置
        best_retrieval = None
        best_retrieval_score = -1
        best_quality = None
        best_quality_score = -1
        
        for exp_name, data in self.results.items():
            if "error" in data:
                continue
            
            # 检索性能
            retrieval_score = data["results"]["retrieval_metrics"]["mean_recall"]
            if retrieval_score > best_retrieval_score:
                best_retrieval_score = retrieval_score
                best_retrieval = exp_name
            
            # 答案质量
            quality_score = data["results"]["answer_quality"]["overall_score"]
            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_quality = exp_name
        
        report += f"### 最佳配置推荐\n\n"
        report += f"- **最佳检索性能**: {best_retrieval} (平均召回率: {best_retrieval_score:.4f})\n"
        report += f"- **最佳答案质量**: {best_quality} (整体质量: {best_quality_score:.4f})\n\n"
        
        # 分析不同组件的影响
        report += "### 组件影响分析\n\n"
        
        # 分析检索器类型影响
        report += "#### 检索器类型比较\n\n"
        retriever_results = {}
        for exp_name, data in self.results.items():
            if "error" in data:
                continue
            
            retriever_type = data["config"]["retriever_type"]
            if retriever_type not in retriever_results:
                retriever_results[retriever_type] = []
            
            retriever_results[retriever_type].append({
                "name": exp_name,
                "retrieval": data["results"]["retrieval_metrics"]["mean_recall"],
                "quality": data["results"]["answer_quality"]["overall_score"]
            })
        
        for retriever_type, results in retriever_results.items():
            avg_retrieval = np.mean([r["retrieval"] for r in results])
            avg_quality = np.mean([r["quality"] for r in results])
            report += f"- **{retriever_type}**: 平均召回率 {avg_retrieval:.4f}, 平均质量 {avg_quality:.4f}\n"
        
        # 分析嵌入模型影响（仅对密集检索器）
        report += "\n#### 嵌入模型比较（密集检索器）\n\n"
        embed_model_results = {}
        for exp_name, data in self.results.items():
            if "error" in data or data["config"]["retriever_type"] != "dense":
                continue
            
            embed_model = data["config"]["embed_model"]
            if embed_model not in embed_model_results:
                embed_model_results[embed_model] = []
            
            embed_model_results[embed_model].append({
                "name": exp_name,
                "retrieval": data["results"]["retrieval_metrics"]["mean_recall"],
                "quality": data["results"]["answer_quality"]["overall_score"]
            })
        
        for model, results in embed_model_results.items():
            avg_retrieval = np.mean([r["retrieval"] for r in results])
            avg_quality = np.mean([r["quality"] for r in results])
            report += f"- **{model}**: 平均召回率 {avg_retrieval:.4f}, 平均质量 {avg_quality:.4f}\n"
        
        # 分析提示模板影响
        report += "\n#### 提示模板比较\n\n"
        prompt_results = {}
        for exp_name, data in self.results.items():
            if "error" in data:
                continue
            
            prompt_type = data["config"]["prompt_type"]
            if prompt_type not in prompt_results:
                prompt_results[prompt_type] = []
            
            prompt_results[prompt_type].append({
                "name": exp_name,
                "faithfulness": data["results"]["answer_quality"]["avg_faithfulness"],
                "relevance": data["results"]["answer_quality"]["avg_relevance"],
                "completeness": data["results"]["answer_quality"]["avg_completeness"],
                "f1": data["results"]["answer_quality"]["avg_f1"],
                "overall": data["results"]["answer_quality"]["overall_score"]
            })
        
        for prompt_type, results in prompt_results.items():
            avg_faithfulness = np.mean([r["faithfulness"] for r in results])
            avg_relevance = np.mean([r["relevance"] for r in results])
            avg_completeness = np.mean([r["completeness"] for r in results])
            avg_f1 = np.mean([r["f1"] for r in results])
            avg_overall = np.mean([r["overall"] for r in results])
            
            report += f"- **{prompt_type}**: 忠实度 {avg_faithfulness:.4f}, 相关性 {avg_relevance:.4f}, "
            report += f"完整性 {avg_completeness:.4f}, F1 {avg_f1:.4f}, 整体 {avg_overall:.4f}\n"
        
        report += "\n### 实验配置详情\n\n"
        for exp_name, data in self.results.items():
            report += f"#### {exp_name}\n\n"
            report += f"- 检索器类型: {data['config']['retriever_type']}\n"
            if data['config']['retriever_type'] == "dense":
                report += f"- 嵌入模型: {data['config']['embed_model']}\n"
            report += f"- 提示模板: {data['config']['prompt_type']}\n\n"
        
        return report


def main():
    """主函数 - 运行比较实验"""
    print("开始RAG系统比较实验...")
    
    # 初始化比较实验
    experiment = ComparisonExperiment()
    
    # 运行所有实验
    print("运行比较实验...")
    results = experiment.run_comparison_experiments()
    
    print("比较实验完成！")
    
    # 显示简要结果
    successful_experiments = [name for name, data in results.items() if "error" not in data]
    failed_experiments = [name for name, data in results.items() if "error" in data]
    
    print(f"\n实验完成情况:")
    print(f"- 成功: {len(successful_experiments)} 个实验")
    print(f"- 失败: {len(failed_experiments)} 个实验")
    
    if failed_experiments:
        print(f"失败的实验: {', '.join(failed_experiments)}")


if __name__ == "__main__":
    main()