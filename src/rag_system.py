import logging
import os
from pathlib import Path
from data_loader import load_and_parse_jsonl
from processor import process_and_save_chunks
from embedder import ChunkEmbedder
from retriever import VectorRetriever
from generator import RAGGenerator


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
        base_url: str = None
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

        logger.info("RAG 系统初始化完成。")

    def query(self, question: str, top_k=5):
        logger.info(f"用户问题: {question}")

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
    # === 路径 ===
    BASE = "C:\\Users\\Wong\\Desktop\\work\\stardew-rag-project\\data"

    INPUT_JSON_PATH = f"{BASE}\\rag_docs.json"
    CHUNKS_PATH = f"{BASE}\\chunks.jsonl"
    METADATA_PATH = f"{BASE}\\chunk_metadata.jsonl"
    EMBEDDINGS_PATH = f"{BASE}\\embeddings.npy"
    INDEX_PATH = f"{BASE}\\faiss_index.bin"

    # === 模型名称 ===
    EMBED_MODEL = "moka-ai/m3e-base"
    LLM_MODEL = "gpt-4-turbo"

    # === API ===
    API_KEY = "sk-tT5HcopxjJ7vGdnX4333Ef20D1E44eB7827b98D4A923F9E2"
    BASE_URL = "https://bj.yi-zhan.top/v1"

    # === Step 1: 打印前 5 条原文（可选） ===
    print("开始加载 JSON 数据...")
    for i, (title, content) in enumerate(load_and_parse_jsonl(INPUT_JSON_PATH), start=1):
        print(f"\n--- 第{i}条 ---")
        print("标题:", title)
        print("内容预览:", content[:200])
        if i >= 5:
            break

    # === Step 2: 分块 ===
    print("\n开始分块...")
    process_and_save_chunks(
        input_file_path=INPUT_JSON_PATH,
        output_file_path=CHUNKS_PATH,
        chunk_size=1024,
        overlap=50
    )

    # === Step 3: 生成嵌入 ===
    print("\n开始生成嵌入...")
    embedder = ChunkEmbedder(model_name=EMBED_MODEL)
    embedder.embed_chunks(
        chunks_file_path=CHUNKS_PATH,
        output_embeddings_path=EMBEDDINGS_PATH,
        output_metadata_path=METADATA_PATH
    )

    # === Step 4: 初始化完整 RAG ===
    rag = RAGSystem(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        index_path=INDEX_PATH,
        model_name=EMBED_MODEL,
        llm_model_name=LLM_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # === Step 5: 测试查询 ===
    test1 = "枫糖浆是什么？"
    print(f"\n问题: {test1}")
    print("回答:", rag.query(test1, top_k=3))

    test2 = "矿车怎么用？"
    print(f"\n问题: {test2}")
    print("回答:", rag.query(test2, top_k=3))
