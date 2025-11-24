import logging
from typing import Optional
from pathlib import Path
from data_loader import load_and_parse_jsonl
from processor import process_and_save_chunks
from embedder import ChunkEmbedder
from retriever import VectorRetriever
from generator import RAGGenerator
from query_rewriter import QueryRewriter
from config import load_config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(
        self,
        embeddings_path: str,
        metadata_path: str,
        index_path: str,
        model_name: str,
        llm_model_name: str,
        api_key: str = None,
        base_url: str = None,
        query_rewrite_mode: str = "none",
        query_rewrite_model: Optional[str] = None
    ):
        logger.info("正在初始化 RAG 系统...")

        self.embedder = ChunkEmbedder(model_name=model_name)

        self.retriever = VectorRetriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            index_path=index_path
        )

        logger.info("正在加载生成模型...")
        self.generator = RAGGenerator(api_key=api_key, model_name=llm_model_name, base_url=base_url)

        # Optional query rewriting / HyDE
        self.query_rewrite_mode = (query_rewrite_mode or "none").lower()
        self.query_rewriter = None
        if self.query_rewrite_mode != "none":
            try:
                self.query_rewriter = QueryRewriter(
                    api_key=api_key,
                    model_name=query_rewrite_model or llm_model_name,
                    base_url=base_url,
                    mode=self.query_rewrite_mode
                )
                logger.info(f"已开启查询改写模式: {self.query_rewrite_mode}")
            except Exception as e:
                logger.warning(f"初始化 QueryRewriter 失败，将使用原始查询。错误: {e}")
                self.query_rewriter = None

        logger.info("RAG 系统初始化完成。")

    def query(self, question: str, top_k=5):
        logger.info(f"用户问题: {question}")

        # 执行查询改写或 HyDE
        rewritten_text = question
        mode_used = "none"
        if self.query_rewriter:
            rewritten_text, mode_used = self.query_rewriter.rewrite(question)
            logger.info(f"查询改写模式={mode_used}，改写结果预览: {rewritten_text[:120]}")
        else:
            logger.info("查询改写未启用。")

        # HyDE 返回的是假设文档；condense/none 返回问题本身
        try:
            q_emb = self.embedder.embed_single_text(rewritten_text)
        except Exception as e:
            logger.warning(f"改写文本嵌入失败，回退使用原始问题。错误: {e}")
            q_emb = self.embedder.embed_single_text(question)

        retrieved = self.retriever.search(q_emb, k=top_k)

        chunks = [meta for meta, score in retrieved]

        if not chunks:
            return "未找到相关内容。"

        print("\n=== 检索到的内容 ===")
        for c in chunks:
            print(c["title"], c["text"][:150])
        print("=====================\n")

        answer = self.generator.generate_answer(question, chunks)
        return answer


if __name__ == "__main__":
    paths, models, pipeline = load_config()

    INPUT_JSON_PATH = paths.raw_docs
    CHUNKS_PATH = paths.chunks
    METADATA_PATH = paths.chunk_metadata
    EMBEDDINGS_PATH = paths.embeddings
    INDEX_PATH = paths.faiss_index

    EMBED_MODEL = models.embed_model
    LLM_MODEL = models.llm_model
    API_KEY = models.openai_api_key
    BASE_URL = models.openai_base_url

    # === Step 1: 打印前 5 条原文（可选） ===
    print("开始加载 JSON 数据...")
    for i, (title, content) in enumerate(load_and_parse_jsonl(str(INPUT_JSON_PATH)), start=1):
        print(f"\n--- 第{i}条 ---")
        print("标题:", title)
        print("内容预览:", content[:200])
        if i >= 5:
            break

    # === Step 2: 分块 ===
    print("\n开始分块...")
    process_and_save_chunks(
        input_file_path=str(INPUT_JSON_PATH),
        output_file_path=str(CHUNKS_PATH),
        chunk_size=pipeline.chunk_size,
        overlap=pipeline.chunk_overlap
    )

    # === Step 3: 生成嵌入 ===
    print("\n开始生成嵌入...")
    embedder = ChunkEmbedder(model_name=EMBED_MODEL)
    embedder.embed_chunks(
        chunks_file_path=str(CHUNKS_PATH),
        output_embeddings_path=str(EMBEDDINGS_PATH),
        output_metadata_path=str(METADATA_PATH)
    )

    # === Step 4: 初始化完整 RAG ===
    rag = RAGSystem(
        embeddings_path=str(EMBEDDINGS_PATH),
        metadata_path=str(METADATA_PATH),
        index_path=str(INDEX_PATH),
        model_name=EMBED_MODEL,
        llm_model_name=LLM_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        query_rewrite_mode=models.query_rewrite_mode,
        query_rewrite_model=models.query_rewrite_model or models.llm_model
    )

    # === Step 5: 测试查询 ===
    test1 = "枫糖浆是什么？"
    print(f"\n问题: {test1}")
    print("回答:", rag.query(test1, top_k=pipeline.top_k))

    test2 = "矿车怎么用？"
    print(f"\n问题: {test2}")
    print("回答:", rag.query(test2, top_k=pipeline.top_k))

    from interactive_interface import run_interactive_query
    run_interactive_query(rag)
