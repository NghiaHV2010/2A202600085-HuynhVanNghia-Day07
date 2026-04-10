from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    ChunkingStrategyComparator,
    Document,
    EMBEDDING_PROVIDER_ENV,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    LocalEmbedder,
    OPENAI_EMBEDDING_MODEL,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)

DOC_DIR = ROOT / "data/yte_md"
METADATA_PATH = ROOT / "strategy/metadata.json"

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Bệnh Alzheimer được chẩn đoán chia giai đoạn theo thang điểm MMSE như thế nào?",
        "gold_answer": "Alzheimer nhẹ: MMSE 21-26, trung bình: 10-20, trung bình nặng: 10-14, nặng: dưới 10.",
        "gold_doc_id": "alzheimer",
        "metadata_filter": {"category": "Thần kinh"},
    },
    {
        "id": 2,
        "query": "Các biện pháp cận lâm sàng nào giúp đánh giá tình trạng mạch máu và tưới máu bàn chân ở người bệnh đái tháo đường?",
        "gold_answer": "Sử dụng các biện pháp đo chỉ số ABI (Ankle Brachial Index) và đo TcPO2 (transcutaneous oxygen tension).",
        "gold_doc_id": "ban-chan-dai-thao-duong",
        "metadata_filter": {"category": "Nội tiết - Đái tháo đường"},
    },
    {
        "id": 3,
        "query": "Kháng sinh điều trị cho bệnh nhân bị khối áp xe phổi nguyên phát thường phải kéo dài trong khoảng thời gian bao lâu?",
        "gold_answer": "Thời gian dùng kháng sinh kéo dài từ 4 đến 6 tuần hoặc cho đến khi X-quang ngực sạch hoặc chỉ còn vết sẹo nhỏ.",
        "gold_doc_id": "ap-xe-phoi",
        "metadata_filter": {"category": "Hô hấp"},
    },
    {
        "id": 4,
        "query": "Nguyên nhân hàng đầu (chiếm tỷ lệ cao nhất) gây ra rủi ro băng huyết sau sinh là gì?",
        "gold_answer": "Nguyên nhân hàng đầu gây ra là do tử cung co hồi kém (đờ tử cung).",
        "gold_doc_id": "bang-huyet-sau-sinh",
        "metadata_filter": {"category": "Sản phụ khoa"},
    },
    {
        "id": 5,
        "query": "Tôi thường xuyên bị ăn không tiêu kéo dài, đó có thể là dấu hiệu cảnh báo của những bệnh lý tiêu hóa nguy hiểm nào?",
        "gold_answer": "Viêm loét dạ dày, trào ngược dạ dày thực quản, liệt dạ dày, thoát vị hoành, sỏi mật, viêm tụy, hội chứng ruột kích thích, hoặc ung thư dạ dày.",
        "gold_doc_id": "an-khong-tieu",
        "metadata_filter": {"category": "Tiêu hóa - Gan mật"},
    },
]

FILTER_EVAL_QUERIES = [
    {"id": 1, "query": "Bệnh Alzheimer có di truyền không?", "expected_category": "Thần kinh"},
    {"id": 2, "query": "Nguyên nhân gây ăn không tiêu là gì?", "expected_category": "Tiêu hóa - Gan mật"},
    {"id": 3, "query": "Áp xe hậu môn có tự khỏi không?", "expected_category": "Tiêu hóa - Hậu môn trực tràng"},
    {"id": 4, "query": "Triệu chứng của áp xe phổi?", "expected_category": "Hô hấp"},
    {"id": 5, "query": "Bàn chân đái tháo đường chăm sóc thế nào?", "expected_category": "Nội tiết - Đái tháo đường"},
    {"id": 6, "query": "Băng huyết sau sinh xử lý ra sao?", "expected_category": "Sản phụ khoa"},
    {"id": 7, "query": "Bàng quang tăng hoạt là gì?", "expected_category": "Tiết niệu"},
]

CATEGORY_HINTS = {
    "Thần kinh": ["alzheimer", "mmse", "sa sut tri tue", "than kinh"],
    "Tiêu hóa - Gan mật": ["an khong tieu", "da day", "tieu hoa", "gan mat"],
    "Tiêu hóa - Hậu môn trực tràng": ["ap xe hau mon", "hau mon", "truc trang"],
    "Hô hấp": ["ap xe phoi", "phoi", "ho hap"],
    "Nội tiết - Đái tháo đường": ["dai thao duong", "tieu duong", "ban chan", "abi", "tcpo2"],
    "Sản phụ khoa": ["bang huyet", "sau sinh", "san phu"],
    "Tiết niệu": ["bang quang", "tiet nieu", "tieu gap"],
}


def ensure_utf8_stdout() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def normalize_for_match(text: str) -> str:
    lowered = text.strip().lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(without_marks.split())


def auto_detect_category(query: str, available_categories: list[str]) -> str | None:
    normalized_query = normalize_for_match(query)
    best_category: str | None = None
    best_score = 0

    for category in available_categories:
        score = 0
        category_normalized = normalize_for_match(category)
        if category_normalized and category_normalized in normalized_query:
            score += max(1, len(category_normalized.split()))

        for hint in CATEGORY_HINTS.get(category, []):
            hint_normalized = normalize_for_match(hint)
            if hint_normalized and hint_normalized in normalized_query:
                score += max(1, len(hint_normalized.split()))

        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def truncate_label(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width]


def format_result_cell(result: dict | None, expected_category: str) -> str:
    if result is None:
        return "-"

    metadata = result.get("metadata", {})
    category = str(metadata.get("category", "khac"))
    score = float(result.get("score", 0.0))
    marker = "v" if category == expected_category else "x"
    category_text = truncate_label(category, 22).ljust(22)
    return f"{marker} {category_text} score={score:.4f}"


def precision_for_category(results: list[dict], category: str) -> float:
    if not results:
        return 0.0
    hits = sum(1 for item in results if item.get("metadata", {}).get("category") == category)
    return hits / len(results)


def run_metadata_filter_evaluation(docs: list[Document], embedding_fn, top_k: int = 3) -> None:
    chunker = SentenceChunker(max_sentences_per_chunk=4)
    chunked_docs = chunk_documents(docs, chunker)
    store = EmbeddingStore(collection_name="group_sentence_filter_eval", embedding_fn=embedding_fn)
    store.add_documents(chunked_docs)

    available_categories = sorted({doc.metadata.get("category", "khac") for doc in docs})
    separator = "─" * 100

    print("METADATA FILTER EVALUATION: search() vs search_with_filter()")
    print()

    precision_no_filter_values: list[float] = []
    precision_filtered_values: list[float] = []

    for item in FILTER_EVAL_QUERIES:
        query_id = item["id"]
        query = item["query"]
        expected_category = item["expected_category"]
        auto_category = auto_detect_category(query, available_categories) or expected_category

        results_no_filter = store.search(query, top_k=top_k)
        results_filtered = store.search_with_filter(
            query,
            top_k=top_k,
            metadata_filter={"category": auto_category},
        )

        print(separator)
        print(f"  Q{query_id}: {query}")
        print(f"  Expected category: {expected_category}")
        print(f"  Auto-detected filter: {auto_category}")
        print()
        print("  Rank  search() — no filter                            search_with_filter()")
        print("  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────")

        max_rank = max(len(results_no_filter), len(results_filtered), top_k)
        for rank in range(1, max_rank + 1):
            left = format_result_cell(results_no_filter[rank - 1] if rank - 1 < len(results_no_filter) else None, expected_category)
            right = format_result_cell(results_filtered[rank - 1] if rank - 1 < len(results_filtered) else None, expected_category)
            print(f"  [{rank}]   {left:<47} {right:<47}")

        precision_no_filter = precision_for_category(results_no_filter, expected_category)
        precision_filtered = precision_for_category(results_filtered, expected_category)
        overlap = len({item["id"] for item in results_no_filter} & {item["id"] for item in results_filtered})

        precision_no_filter_values.append(precision_no_filter)
        precision_filtered_values.append(precision_filtered)

        print()
        print(
            "  Precision (no filter): {0:.0f}%   |   Precision (filtered): {1:.0f}%   |   Overlap: {2}/{3}".format(
                precision_no_filter * 100,
                precision_filtered * 100,
                overlap,
                top_k,
            )
        )
        print()

    avg_no_filter = (sum(precision_no_filter_values) / len(precision_no_filter_values) * 100) if precision_no_filter_values else 0.0
    avg_filtered = (sum(precision_filtered_values) / len(precision_filtered_values) * 100) if precision_filtered_values else 0.0
    improvement = avg_filtered - avg_no_filter

    print(separator)
    print(f"  Total queries evaluated:         {len(FILTER_EVAL_QUERIES)}")
    print(f"  Avg precision (no filter):       {avg_no_filter:.1f}%")
    print(f"  Avg precision (with filter):     {avg_filtered:.1f}%")
    print(f"  Precision improvement:           {improvement:+.1f}%")


def load_medical_metadata() -> dict[str, dict[str, str]]:
    raw_text = METADATA_PATH.read_text(encoding="utf-8").strip()
    if raw_text.startswith("MEDICAL_DOCS_METADATA"):
        _, value = raw_text.split("=", maxsplit=1)
        data = ast.literal_eval(value.strip())
    else:
        data = json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError("Medical metadata must be a dictionary")

    return data


def load_documents(metadata_by_doc: dict[str, dict[str, str]]) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(DOC_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        doc_id = path.stem
        medical_meta = metadata_by_doc.get(doc_id, {})
        docs.append(
            Document(
                id=doc_id,
                content=content,
                metadata={
                    "source_file": str(path).replace("\\", "/"),
                    "source": medical_meta.get("source", "unknown"),
                    "disease_name": medical_meta.get("disease_name", doc_id),
                    "category": medical_meta.get("category", "khac"),
                    "format": "md",
                    "language": "vi",
                },
            )
        )
    return docs


def chunk_documents(docs: list[Document], chunker: object) -> list[Document]:
    chunked_docs: list[Document] = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for index, chunk_text in enumerate(chunks):
            text = chunk_text.strip()
            if not text:
                continue
            metadata = dict(doc.metadata)
            metadata.update({"doc_id": doc.id, "chunk_index": index})
            chunked_docs.append(
                Document(
                    id=f"{doc.id}_chunk_{index}",
                    content=text,
                    metadata=metadata,
                )
            )
    return chunked_docs


def safe_summary(text: str, max_chars: int = 140) -> str:
    one_line = " ".join(text.split())
    return one_line[:max_chars] + ("..." if len(one_line) > max_chars else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentence chunking strategy benchmark and QA demo")
    parser.add_argument(
        "--mode",
        choices=["json", "log", "filter-eval"],
        default=os.getenv("STRATEGY_OUTPUT_MODE", "json"),
        help="json: benchmark output as JSON, log: one question log format, filter-eval: screen comparison search vs filtered",
    )
    parser.add_argument("--query", help="Question to run in log mode")
    parser.add_argument("--query-id", type=int, help="Use a benchmark query id in log mode")
    parser.add_argument("--category", help="Metadata category filter in log mode")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks to retrieve in log mode")
    parser.add_argument("--eval-top-k", type=int, default=3, help="Top-k used by filter-eval mode")
    parser.add_argument(
        "--output-file",
        help="Optional output path for JSON mode; writes UTF-8 JSON directly to file",
    )
    return parser.parse_args()


def build_embedder_from_env():
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "openai").strip().lower()
    if provider == "openai":
        openai_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
        try:
            embedder = OpenAIEmbedder(model_name=openai_model_name)
            embedding_backend = getattr(embedder, "_backend_name", openai_model_name)
        except Exception as exc:
            embedder = _mock_embed
            embedding_backend = f"{getattr(embedder, '_backend_name', 'mock')} (openai_init_failed:{exc.__class__.__name__})"
    elif provider == "local":
        local_model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            embedder = LocalEmbedder(model_name=local_model_name)
            embedding_backend = getattr(embedder, "_backend_name", local_model_name)
        except Exception:
            embedder = _mock_embed
            embedding_backend = getattr(embedder, "_backend_name", "mock")
    else:
        embedder = _mock_embed
        embedding_backend = getattr(embedder, "_backend_name", "mock")

    return provider, embedder, embedding_backend


def resolve_log_query(args: argparse.Namespace) -> dict:
    selected: dict | None = None
    if args.query_id is not None:
        selected = next((item for item in BENCHMARK_QUERIES if item["id"] == args.query_id), None)
    elif args.query:
        normalized_query = args.query.strip().lower()
        selected = next((item for item in BENCHMARK_QUERIES if item["query"].strip().lower() == normalized_query), None)

    if selected is None:
        if args.query:
            selected = {
                "id": 0,
                "query": args.query.strip(),
                "gold_answer": "",
                "gold_doc_id": "",
                "metadata_filter": None,
            }
        else:
            selected = BENCHMARK_QUERIES[-1]

    query_item = dict(selected)
    if args.category:
        query_item["metadata_filter"] = {"category": args.category}

    return query_item


def print_log_demo(
    docs: list[Document],
    embedding_fn,
    query_item: dict,
    top_k: int,
) -> None:
    chunker = SentenceChunker(max_sentences_per_chunk=4)
    chunked_docs = chunk_documents(docs, chunker)
    store = EmbeddingStore(collection_name="group_sentence_live_demo", embedding_fn=embedding_fn)
    store.add_documents(chunked_docs)

    llm_fn, _ = build_llm_fn()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    question = query_item["query"]
    metadata_filter = query_item.get("metadata_filter")

    if metadata_filter:
        results = store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
    else:
        results = store.search(question, top_k=top_k)

    metadata_path = str(METADATA_PATH.relative_to(ROOT)).replace("\\", "/")
    print(f"Metadata file: {metadata_path}")
    print(f"Question: {question}")
    if metadata_filter:
        filter_text = ", ".join(f"{key} = {value}" for key, value in metadata_filter.items())
        print(f"Metadata filter: {filter_text}")

    print("\nRetrieving relevant chunks ...")
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source_file = metadata.get("source_file") or f"{metadata.get('doc_id', 'unknown')}.md"
        source_name = Path(source_file).name
        preview = safe_summary(result.get("content", ""), max_chars=150)
        print(f"  [{index}] ({source_name}, score={float(result.get('score', 0.0)):.4f}) {preview}")

    print("\nGenerating answer ...\n")
    answer = agent.answer(question=question, top_k=top_k, metadata_filter=metadata_filter)
    print("Answer:")
    print(answer)


def write_json_output(output: dict, output_file: str) -> None:
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to: {output_path}")


def fallback_llm_from_context(prompt: str) -> str:
    try:
        context_part = prompt.split("Context:\n", maxsplit=1)[1].split("\n\nQuestion:", maxsplit=1)[0]
    except Exception:
        return "Mình chưa có đủ thông tin từ ngữ cảnh để trả lời chính xác."

    lines = []
    for line in context_part.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[Chunk "):
            continue
        if stripped.startswith("Metadata filter:"):
            continue
        lines.append(stripped)

    if not lines:
        return "Mình chưa có đủ thông tin từ ngữ cảnh để trả lời chính xác."

    text = " ".join(lines)
    return text[:420] + ("..." if len(text) > 420 else "")


def build_llm_fn() -> tuple[Callable[[str], str], str]:
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if llm_provider != "openai":
        return fallback_llm_from_context, "context-fallback"

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    try:
        from openai import OpenAI

        client = OpenAI()
    except Exception:
        return fallback_llm_from_context, "context-fallback"

    def _openai_llm(prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model=chat_model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý y khoa. Trả lời ngắn gọn bằng tiếng Việt, chỉ dựa trên context được cung cấp.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            cleaned = content.strip()
            return cleaned or fallback_llm_from_context(prompt)
        except Exception:
            return fallback_llm_from_context(prompt)

    return _openai_llm, chat_model


def run_retrieval(
    docs: list[Document],
    chunker: object,
    embedding_fn,
    collection_name: str,
) -> dict:
    chunked_docs = chunk_documents(docs, chunker)
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedding_fn)
    store.add_documents(chunked_docs)

    rows = []
    relevant_top3 = 0

    for item in BENCHMARK_QUERIES:
        metadata_filter = item["metadata_filter"]
        if metadata_filter:
            results = store.search_with_filter(item["query"], top_k=3, metadata_filter=metadata_filter)
        else:
            results = store.search(item["query"], top_k=3)

        top_doc_ids = [result.get("metadata", {}).get("doc_id") for result in results]
        is_relevant = item["gold_doc_id"] in top_doc_ids
        relevant_top3 += int(is_relevant)

        top1 = results[0] if results else {"content": "", "score": 0.0, "metadata": {}}
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "gold_doc_id": item["gold_doc_id"],
                "metadata_filter": metadata_filter,
                "top_doc_ids": top_doc_ids,
                "relevant_top3": is_relevant,
                "top1_score": round(float(top1.get("score", 0.0)), 4),
                "top1_doc_id": top1.get("metadata", {}).get("doc_id"),
                "top1_chunk_summary": safe_summary(top1.get("content", "")),
            }
        )

    avg_chunk_length = mean(len(doc.content) for doc in chunked_docs) if chunked_docs else 0.0

    return {
        "chunk_count": len(chunked_docs),
        "avg_chunk_length": round(avg_chunk_length, 2),
        "relevant_top3": relevant_top3,
        "rows": rows,
    }


def run_agent_qa_examples(docs: list[Document], embedding_fn) -> dict:
    chunker = SentenceChunker(max_sentences_per_chunk=4)
    chunked_docs = chunk_documents(docs, chunker)
    store = EmbeddingStore(collection_name="group_sentence_agent_qa", embedding_fn=embedding_fn)
    store.add_documents(chunked_docs)

    llm_fn, llm_backend = build_llm_fn()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)

    rows = []
    for item in BENCHMARK_QUERIES:
        metadata_filter = item.get("metadata_filter")
        if metadata_filter:
            retrieved = store.search_with_filter(item["query"], top_k=3, metadata_filter=metadata_filter)
        else:
            retrieved = store.search(item["query"], top_k=3)

        top_doc_ids = [result.get("metadata", {}).get("doc_id") for result in retrieved]
        answer = agent.answer(
            question=item["query"],
            top_k=3,
            metadata_filter=metadata_filter,
        )
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "gold_answer": item["gold_answer"],
                "metadata_filter": metadata_filter,
                "top_doc_ids": top_doc_ids,
                "top1_chunk_summary": safe_summary(retrieved[0]["content"]) if retrieved else "",
                "agent_answer": answer,
            }
        )

    return {
        "llm_backend": llm_backend,
        "rows": rows,
    }


def main() -> None:
    load_dotenv(override=False)
    ensure_utf8_stdout()
    args = parse_args()

    metadata_by_doc = load_medical_metadata()
    docs = load_documents(metadata_by_doc)

    provider, embedder, embedding_backend = build_embedder_from_env()

    if args.mode == "log":
        query_item = resolve_log_query(args)
        print_log_demo(
            docs=docs,
            embedding_fn=embedder,
            query_item=query_item,
            top_k=max(1, args.top_k),
        )
        return

    if args.mode == "filter-eval":
        run_metadata_filter_evaluation(
            docs=docs,
            embedding_fn=embedder,
            top_k=max(1, args.eval_top_k),
        )
        return

    doc_inventory = [
        {
            "doc_id": doc.id,
            "disease_name": doc.metadata["disease_name"],
            "category": doc.metadata["category"],
            "source": doc.metadata["source"],
            "source_file": doc.metadata["source_file"],
            "char_count": len(doc.content),
        }
        for doc in docs
    ]

    selected_for_baseline = ["alzheimer", "ap-xe-phoi", "bang-quang-tang-hoat"]
    comparator = ChunkingStrategyComparator()
    baseline = {}
    for doc_id in selected_for_baseline:
        text = next(doc.content for doc in docs if doc.id == doc_id)
        compare_data = comparator.compare(text, chunk_size=500)
        baseline[doc_id] = {
            strategy: {
                "count": values["count"],
                "avg_length": round(float(values["avg_length"]), 2),
            }
            for strategy, values in compare_data.items()
        }

    retrieval_fixed = run_retrieval(
        docs,
        chunker=FixedSizeChunker(chunk_size=500, overlap=50),
        embedding_fn=embedder,
        collection_name="group_fixed_strategy",
    )
    retrieval_sentence = run_retrieval(
        docs,
        chunker=SentenceChunker(max_sentences_per_chunk=4),
        embedding_fn=embedder,
        collection_name="group_sentence_strategy",
    )
    retrieval_recursive = run_retrieval(
        docs,
        chunker=RecursiveChunker(chunk_size=500),
        embedding_fn=embedder,
        collection_name="group_recursive_strategy",
    )
    agent_qa = run_agent_qa_examples(docs, embedding_fn=embedder)

    output = {
        "metadata_path": str(METADATA_PATH.relative_to(ROOT)).replace("\\", "/"),
        "embedding_provider": provider,
        "embedding_backend": embedding_backend,
        "llm_backend": agent_qa["llm_backend"],
        "doc_inventory": doc_inventory,
        "baseline": baseline,
        "benchmark_queries": BENCHMARK_QUERIES,
        "retrieval_summary": {
            "fixed_size": {
                "chunk_count": retrieval_fixed["chunk_count"],
                "avg_chunk_length": retrieval_fixed["avg_chunk_length"],
                "relevant_top3": retrieval_fixed["relevant_top3"],
            },
            "sentence_chunker": {
                "chunk_count": retrieval_sentence["chunk_count"],
                "avg_chunk_length": retrieval_sentence["avg_chunk_length"],
                "relevant_top3": retrieval_sentence["relevant_top3"],
            },
            "recursive": {
                "chunk_count": retrieval_recursive["chunk_count"],
                "avg_chunk_length": retrieval_recursive["avg_chunk_length"],
                "relevant_top3": retrieval_recursive["relevant_top3"],
            },
        },
        "sentence_rows": retrieval_sentence["rows"],
        "agent_qa_rows": agent_qa["rows"],
    }

    if args.output_file:
        write_json_output(output, args.output_file)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
