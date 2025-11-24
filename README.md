# 星露谷物语 RAG Project

一个面向《星露谷物语》中文资料的检索增强生成（RAG）实验项目，包含数据清洗与分块、向量/稀疏检索、答案生成、评估以及多配置对比实验。

## 功能特性
- JSON/JSONL 数据流式加载与分块（支持自定义块大小与重叠）
- 支持密集检索（FAISS + sentence-transformers 嵌入）与稀疏检索（BM25）
- 可切换的提示模板（标准、详细、星露谷专用、极简）
- OpenAI 兼容接口的答案生成
- 自动构建评估集与完整的检索/生成质量评估
- 多模型、多提示、多检索器的对比实验报告

## 目录结构
- `src/`
  - `rag_system.py`：端到端 RAG 流程示例（分块、嵌入、索引、查询、交互式问答）
  - `processor.py`/`chunker.py`/`data_loader.py`：数据加载、分块与持久化
  - `embedder.py`：使用 sentence-transformers 生成嵌入
  - `retriever.py` / `sparse_retriever.py`：FAISS 向量检索与 BM25 稀疏检索
  - `generator.py`：基于 OpenAI 兼容接口的答案生成与多提示模板
  - `rag_evaluator.py` / `evaluation_set_creator.py`：评估集构建与检索/生成指标评估
  - `comparison_experiment.py`：多配置对比实验与报告生成
- `data/`：存放原始文档、分块、嵌入、索引与评估结果（需自行准备）
- `requirements.txt`：主要依赖

## 环境准备
1. Python 3.9+（建议虚拟环境）
2. 安装依赖：
   ```bash
   pip install -r requirements.txt openai
   ```
   如使用 GPU，可安装对应的 `torch` 版本。
3. 配置 LLM 访问：
   - 设置环境变量 `OPENAI_API_KEY`，或在代码中传入 `api_key`
   - 如需自定义网关，设置 `base_url`（示例中使用了自定义地址）

## 数据准备
- 将原始游戏百科数据放在 `data/rag_docs.json`（支持 JSON 或 JSONL，每条包含 `title` 与 `text` 字段）。
- 运行过程中会生成：
  - `data/chunks.jsonl`：分块后的文本
  - `data/embeddings.npy`：嵌入向量
  - `data/chunk_metadata.jsonl`：与嵌入对应的元数据
  - `data/faiss_index.bin`：可选，保存的 FAISS 索引

## 快速上手
以下命令默认在仓库根目录执行，并使用 `PYTHONPATH=src` 让相对导入可用。

1) **分块并生成嵌入、索引 + 简单查询**
   ```bash
   PYTHONPATH=src OPENAI_API_KEY=your_key \
   python src/rag_system.py
   ```
   脚本会：
   - 打印前几条原始文档
   - 将文档分块并写入 `data/chunks.jsonl`
   - 生成嵌入与 FAISS 索引
   - 对示例问题进行检索和回答
   - 启动交互式问答循环（输入“退出/quit”结束）

2) **独立运行评估**
   ```bash
   PYTHONPATH=src OPENAI_API_KEY=your_key \
   python src/rag_evaluator.py
   ```
   - 若 `data/evaluation_set.json` 不存在，会自动构建评估集
   - 输出检索指标（Recall@K、MRR）与答案质量指标（忠实度、相关性、完整性、F1）
   - 结果与摘要报告保存到 `evaluation_results/`

3) **多配置对比实验**
   ```bash
   PYTHONPATH=src OPENAI_API_KEY=your_key \
   python src/comparison_experiment.py
   ```
   - 预置多组密集/稀疏检索与不同提示模板的组合
   - 自动运行评估并在 `comparison_results/` 下生成汇总表格与详细报告

## 配置说明
- 嵌入模型：默认 `moka-ai/m3e-base`，可切换为 BAAI/bge 系列或英文模型
- 检索方式：向量检索（`VectorRetriever` + FAISS）或 BM25 稀疏检索
- 提示模板：`standard` / `detailed` / `stardew_specific` / `minimal`
- 关键路径：在各脚本顶部的 `BASE`、`*_PATH` 与模型名称可按需修改
- API 配置：`OPENAI_API_KEY` 与 `base_url` 建议通过环境变量或启动参数传入

## 常见问题
- **ImportError（找不到模块）**：确保命令前设置 `PYTHONPATH=src`，或在 IDE 中将 `src` 设为源码根目录。
- **内存/速度**：数据量大时，`embedder.embed_chunks` 可能占用较多内存，可自行拆分批次。
- **缺少依赖**：如遇 `openai` 未安装，执行 `pip install openai`。

欢迎根据自己的数据源、模型与提示风格进一步扩展。若要集成到其他应用，可复用 `RAGSystem`，只需提供已有的嵌入、元数据与索引路径即可。
