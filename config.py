"""Central configuration for the Stardew RAG project.

Import this module in your scripts (e.g. rag_system.py or rag_evaluator.py)
to keep paths and model choices in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Optional

# Project-level directories
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
COMPARISON_RESULTS_DIR = PROJECT_ROOT / "comparison_results"


@dataclass
class PathConfig:
    """File locations used across the pipeline."""

    base_dir: Path = field(default_factory=lambda: DATA_DIR)
    raw_docs: Path = None
    chunks: Path = None
    chunk_metadata: Path = None
    embeddings: Path = None
    faiss_index: Path = None

    def __post_init__(self):
        # Default to the standard filenames under base_dir
        self.raw_docs = self.raw_docs or (self.base_dir / "rag_docs.json")
        self.chunks = self.chunks or (self.base_dir / "chunks.jsonl")
        self.chunk_metadata = self.chunk_metadata or (self.base_dir / "chunk_metadata.jsonl")
        self.embeddings = self.embeddings or (self.base_dir / "embeddings.npy")
        self.faiss_index = self.faiss_index or (self.base_dir / "faiss_index.bin")

    def ensure_dirs(self):
        """Create needed folders if they do not exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model settings with environment-variable overrides."""

    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "moka-ai/m3e-base"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4-turbo"))
    prompt_type: str = field(default_factory=lambda: os.getenv("PROMPT_TYPE", "standard"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://bj.yi-zhan.top/v1"))
    query_rewrite_mode: str = field(default_factory=lambda: os.getenv("QUERY_REWRITE_MODE", "none"))  # none | condense | hyde
    query_rewrite_model: Optional[str] = field(default_factory=lambda: os.getenv("QUERY_REWRITE_MODEL"))  # fallback to llm_model


@dataclass
class PipelineConfig:
    """Processing defaults."""

    chunk_size: int = 1024
    chunk_overlap: int = 50
    top_k: int = 3  # default retrieval size in interactive mode


def load_config() -> tuple[PathConfig, ModelConfig, PipelineConfig]:
    """Convenience helper to get ready-to-use config objects."""
    paths = PathConfig()
    models = ModelConfig()
    pipeline = PipelineConfig()
    paths.ensure_dirs()
    return paths, models, pipeline


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "EVAL_RESULTS_DIR",
    "COMPARISON_RESULTS_DIR",
    "PathConfig",
    "ModelConfig",
    "PipelineConfig",
    "load_config",
]
