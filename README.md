# Stardew Valley RAG Project

## Team Members: 
Guorun WANG (ID: 50012625), 
Leyi SHENG (ID: 50013289), 
Xuqiao MA (ID: 50013100)

Retrieval-augmented generation (RAG) for Chinese Stardew Valley knowledge: data prep and chunking, dense/sparse retrieval, answer generation, evaluation, and configuration comparisons.

## Highlights
- Stream JSON/JSONL, chunk with configurable size/overlap
- Dense retrieval (FAISS + sentence-transformers) and sparse retrieval (BM25)
- Optional query rewriting: `condense` / `hyde` / `none`
- Prompt templates: `standard` / `detailed` / `stardew_specific` / `minimal`
- OpenAI-compatible generation with custom `base_url`
- Automated evaluation (retrieval + answer quality) and multi-config comparison reports

## Environment & Dependencies
1) Python 3.9+ (use a virtualenv).
2) Install dependencies (GPU users install a matching `torch` separately):
   ```bash
   pip install -r requirements.txt openai
   ```
3) LLM access (override the demo key in `config.py` with env vars):
   - `OPENAI_API_KEY=your_key`
   - optional: `OPENAI_BASE_URL=https://your-endpoint/v1`

## Data
- Place source docs at `data/rag_docs.json` (JSON array or JSONL). Each record needs `title` and `text`, for example:
  ```json
  {"title": "枫糖浆 - 星露谷物语官方中文维基", "text": "枫糖浆是一种可在树上使用取蜜器获得的采集品..."}
  ```
- Generated artifacts:
  - `data/chunks.jsonl` – chunks
  - `data/embeddings.npy` – embeddings
  - `data/chunk_metadata.jsonl` – chunk titles/text
  - `data/faiss_index.bin` – FAISS index (dense retrieval)

## Quickstart (run from repo root)
Use `PYTHONPATH=.:src` so local imports work.

1) **End-to-end: chunk + embed + query + interactive**
   ```bash
   PYTHONPATH=.:src OPENAI_API_KEY=your_key \
   python src/rag_system.py
   ```
   Prints sample docs, builds chunks/embeddings/index, answers sample questions, then enters an interactive loop (`退出/quit` to exit).

2) **Build evaluation set**
   ```bash
   PYTHONPATH=.:src python src/evaluation_set_creator.py
   ```
   Writes sampled Q&A to `data/evaluation_set.json` for repeatable evaluation.

3) **Evaluate retrieval and answer quality**
   ```bash
   PYTHONPATH=.:src OPENAI_API_KEY=your_key \
   python src/rag_evaluator.py
   ```
   Reports Recall@K, MRR, faithfulness/relevance/completeness/F1; outputs to `evaluation_results/`.

4) **Run comparison experiments**
   ```bash
   PYTHONPATH=.:src OPENAI_API_KEY=your_key \
   python src/comparison_experiment.py
   ```
   Benchmarks multiple dense/sparse + prompt combinations; results and report in `comparison_results/`.

## Script Cheat Sheet
- `src/rag_system.py`: end-to-end example (chunk, embed, index, QA, interactive loop)
- `src/processor.py` / `src/chunker.py` / `src/data_loader.py`: loading and chunking
- `src/embedder.py`: sentence-transformers embeddings
- `src/retriever.py` / `src/sparse_retriever.py`: FAISS vector retrieval and BM25
- `src/generator.py`: OpenAI-compatible generation with prompt templates
- `src/query_rewriter.py`: condense / HyDE query rewriting
- `src/evaluation_set_creator.py` / `src/rag_evaluator.py`: eval set creation and metrics
- `src/comparison_experiment.py`: multi-model/retriever/prompt comparisons

## Config & Tunables
Centralized in `config.py`, overridable via env vars:
- Models: `EMBED_MODEL` (default `moka-ai/m3e-base`), `LLM_MODEL` (default `gpt-4-turbo`)
- Prompts & rewriting: `PROMPT_TYPE`, `QUERY_REWRITE_MODE`=`none|condense|hyde`, `QUERY_REWRITE_MODEL`
- API: `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- Chunking/retrieval: `chunk_size`, `chunk_overlap`, `top_k` (edit `PipelineConfig`)
- Paths: default `data/`; evaluation in `evaluation_results/`; comparisons in `comparison_results/`

## FAQ
- **ImportError / module not found**: prefix commands with `PYTHONPATH=.:src`, or mark `src` as source root in your IDE.
- **Memory/time during embedding**: `embedder.embed_chunks` encodes all text at once; for large data, batch or reduce chunk size.
- **Missing deps**: `pip install -r requirements.txt openai`; install a GPU build of `torch` if needed.
- **API key hygiene**: the repo contains a demo key in `config.py`—always override via env vars before running or sharing code.

Feel free to swap in your data, models, and prompts. If you already have embeddings and an index, you can plug them into `RAGSystem` to query or integrate elsewhere.***

## Contributions

Guorun WANG: Designed and implemented a fully functional, end-to-end RAG pipeline from scratch and exploration on Latency and Memory Profiling, arranged presentation, revised report.


Leyi SHENG:Crawlled the data from stardew valley wiki, realized the function of query rewriting and add config and readme, finished whole report writing.


Xuqiao MA: Designed a simple use interface. Evaluation set generation method and implemented evaluation and comparison experiments, revised report.

