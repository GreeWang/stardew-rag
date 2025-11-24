import logging
import os
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    LLM-based query rewriting utilities (condense question or HyDE).
    Mode can be "condense", "hyde", or "none".
    """

    def __init__(
        self,
        api_key: Optional[str],
        model_name: str,
        base_url: Optional[str] = None,
        mode: str = "none",
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("QueryRewriter requires OPENAI_API_KEY or api_key parameter.")

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model_name = model_name
        self.mode = mode.lower()

    def condense_question(self, question: str) -> str:
        """Rewrite a possibly vague question into a concise, well-formed one."""
        prompt = (
            "Rewrite the following user question to be concise, specific, and well-formed. "
            "Keep it about the game Stardew Valley. Do not add new facts.\n\n"
            f"User question: {question}\n\n"
            "Rewritten question:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a concise question rewriter for Stardew Valley topics."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=128,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Query rewrite failed (condense), fallback to original: {e}")
            return question

    def generate_hyde_document(self, question: str) -> str:
        """Generate a hypothetical answer passage (HyDE) to use for embedding-based retrieval."""
        prompt = (
            "Write a short, factual passage (80-120 words) that would answer the user's question "
            "about Stardew Valley. This is a hypothetical document used for retrieval; avoid speculation "
            "and stick to likely in-game facts.\n\n"
            f"User question: {question}\n\n"
            "Hypothetical passage:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You write grounded, factual passages about Stardew Valley."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=220,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed, fallback to original question: {e}")
            return question

    def rewrite(self, question: str) -> tuple[str, str]:
        """
        Returns (used_text, mode_used). If mode is unsupported or fails, returns original question with 'none'.
        For HyDE, used_text is the hypothetical passage; for condense, it's the rewritten question.
        """
        if self.mode == "condense":
            rewritten = self.condense_question(question)
            return rewritten, "condense"
        if self.mode == "hyde":
            hyde_doc = self.generate_hyde_document(question)
            return hyde_doc, "hyde"
        return question, "none"
