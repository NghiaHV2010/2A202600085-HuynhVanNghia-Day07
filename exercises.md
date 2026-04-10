# Day 7 — Exercises
## Data Foundations: Embedding & Vector Store | Lab Worksheet

---

## Part 1 — Warm-up (Cá nhân)

### Exercise 1.1 — Cosine Similarity in Plain Language

No math required — explain conceptually:

- What does it mean for two text chunks to have high cosine similarity?
- Give a concrete example of two sentences that would have HIGH similarity and two that would have LOW similarity.
- Why is cosine similarity preferred over Euclidean distance for text embeddings?

> **Ghi kết quả vào:** Report — Section 1 (Warm-up)

---

### Exercise 1.2 — Chunking Math

- A document is 10,000 characters. You chunk it with `chunk_size=500`, `overlap=50`. How many chunks do you expect?
- Formula: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
- If overlap is increased to 100, how does this change the chunk count? Why would you want more overlap?

> **Ghi kết quả vào:** Report — Section 1 (Warm-up)

---

## Part 2 — Core Coding (Cá nhân)

Implement all TODOs in `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được implement sẵn làm ví dụ — đọc kỹ để hiểu pattern trước khi implement phần còn lại.

Run `pytest tests/` to check progress.

### Checklist
- [x] `Document` dataclass — ĐÃ IMPLEMENT SẴN
- [x] `FixedSizeChunker` — ĐÃ IMPLEMENT SẴN
- [ ] `SentenceChunker` — split on sentence boundaries, group into chunks
- [ ] `RecursiveChunker` — try separators in order, recurse on oversized pieces
- [ ] `compute_similarity` — cosine similarity formula with zero-magnitude guard
- [ ] `ChunkingStrategyComparator` — call all three, compute stats
- [ ] `EmbeddingStore.__init__` — initialize store (in-memory or ChromaDB)
- [ ] `EmbeddingStore.add_documents` — embed and store each document
- [ ] `EmbeddingStore.search` — embed query, rank by dot product
- [ ] `EmbeddingStore.get_collection_size` — return count
- [ ] `EmbeddingStore.search_with_filter` — filter by metadata, then search
- [ ] `EmbeddingStore.delete_document` — remove all chunks for a doc_id
- [ ] `KnowledgeBaseAgent.answer` — retrieve + build prompt + call LLM

> **Nộp code:** `src/`
> **Ghi approach vào:** Report — Section 4 (My Approach)

---

## Part 3 — So Sánh Retrieval Strategy (Nhóm)

### Exercise 3.0 — Chuẩn Bị Tài Liệu (Giờ đầu tiên)

Mỗi nhóm chọn một domain và chuẩn bị bộ tài liệu:

**Step 1 — Chọn domain:** FAQ, SOP, policy, docs kỹ thuật, recipes, luật, y tế, v.v.

**Step 2 — Thu thập 5-10 tài liệu.** Lưu dưới dạng `.txt` hoặc `.md` vào thư mục `data/`.

> **Tip chuyển PDF sang Markdown:**
> - `pip install marker-pdf` → `marker_single input.pdf output/` (chất lượng cao, giữ cấu trúc)
> - `pip install pymupdf4llm` → `pymupdf4llm.to_markdown("input.pdf")` (nhanh, đơn giản)
> - Hoặc copy-paste nội dung từ PDF/web vào file `.txt`

Ghi vào bảng:

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

**Step 3 — Thiết kế metadata schema:** Mỗi tài liệu cần ít nhất 2 trường metadata hữu ích (e.g., `category`, `date`, `source`, `language`, `difficulty`).

> **Ghi kết quả vào:** Report — Section 2 (Document Selection)

---

### Exercise 3.1 — Thiết Kế Retrieval Strategy (Mỗi người thử riêng)

Mỗi thành viên **tự chọn strategy riêng** để thử trên cùng bộ tài liệu nhóm.

**Step 1 — Baseline:** Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu. Ghi kết quả.

**Step 2 — Chọn hoặc thiết kế strategy của bạn:**
- Dùng 1 trong 3 built-in strategies với tham số tối ưu, HOẶC
- Thiết kế custom strategy cho domain (ví dụ: chunk by Q&A pairs, by sections, by headers)
- Mỗi thành viên nên thử strategy **khác nhau** để có gì so sánh

```python
class CustomChunker:
    """Your custom chunking strategy for [your domain].

    Design rationale: [explain why this strategy fits your data]
    """

    def chunk(self, text: str) -> list[str]:
        # Your implementation here
        ...
```

**Step 3 — So sánh:** Custom/tuned strategy vs baseline trên cùng tài liệu.

> **Ghi kết quả vào:** Report — Section 3 (Chunking Strategy)

---

### Exercise 3.2 — Chuẩn Bị Benchmark Queries

Mỗi nhóm viết **đúng 5 benchmark queries** kèm **gold answers**.

| # | Query | Gold Answer (câu trả lời đúng) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

**Yêu cầu:**
- Queries phải đa dạng (không hỏi 5 câu giống nhau)
- Gold answers phải cụ thể và có thể verify từ tài liệu
- Ít nhất 1 query yêu cầu metadata filtering để trả lời tốt

> **Ghi kết quả vào:** Report — Section 6 (Results — Benchmark Queries & Gold Answers)

---

### Exercise 3.3 — Cosine Similarity Predictions (Cá nhân)

Call `compute_similarity()` on 5 pairs of sentences. **Before running**, predict which pairs will have highest/lowest similarity. Record your predictions and the actual results. Reflect on what surprised you most.

> **Ghi kết quả vào:** Report — Section 5 (Similarity Predictions)

---

### Exercise 3.4 — Chạy Benchmark & So Sánh Trong Nhóm

**Step 1:** Mỗi thành viên chạy 5 benchmark queries với strategy riêng. Ghi kết quả top-3 cho mỗi query.

**Step 2:** So sánh kết quả trong nhóm:
- Strategy nào cho retrieval tốt nhất? Tại sao?
- Có query nào mà strategy A tốt hơn B nhưng ngược lại ở query khác?
- Metadata filtering có giúp ích không?

**Step 3:** Thảo luận và rút ra bài học — chuẩn bị cho phần demo với các nhóm khác.

> **Ghi kết quả vào:** Report — Section 6 (Results)
> **Gợi ý đánh giá:** xem checklist ngắn trong `README.md` mục **Cách Tự Đánh Giá Kết Quả Retrieval** hoặc chi tiết hơn trong `docs/EVALUATION.md`.

---

### Exercise 3.5 — Failure Analysis

Tìm ít nhất **1 failure case** trong quá trình so sánh. Mô tả:
- Query nào retrieval thất bại?
- Tại sao? (chunk quá nhỏ/lớn, metadata thiếu, query mơ hồ, v.v.)
- Đề xuất cải thiện?

> **Ghi kết quả vào:** Report — Section 7 (What I Learned)
> **Gợi ý:** failure analysis nên tham chiếu các góc nhìn như precision, chunk coherence, metadata utility, và grounding quality.

---

## Submission Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] `src/` updated (cá nhân)
- [ ] Report completed (`report/REPORT.md` — 1 file/sinh viên)
