from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        if metadata_filter:
            retrieved_chunks = self.store.search_with_filter(
                question,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        else:
            retrieved_chunks = self.store.search(question, top_k=top_k)

        if retrieved_chunks:
            context = "\n\n".join(
                f"[Chunk {index} | score={item['score']:.4f}]\n{item['content']}"
                for index, item in enumerate(retrieved_chunks, start=1)
            )
        else:
            context = "(No relevant context found in the knowledge base.)"

        filter_hint = f"\nMetadata filter: {metadata_filter}" if metadata_filter else ""

        prompt = (
            "You are a helpful assistant. Use only the provided context to answer the question.\n"
            "If the context is insufficient, say you do not have enough information.\n\n"
            f"Context:\n{context}{filter_hint}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
