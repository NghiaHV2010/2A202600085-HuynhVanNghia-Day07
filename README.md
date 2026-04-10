# Ngày 7 — Nền Tảng Dữ Liệu: Embedding & Vector Store

---

## Mục Tiêu

Sau lab này, bạn cần có thể:
- Giải thích cosine similarity và dự đoán điểm tương đồng giữa các văn bản
- Triển khai 3 chiến lược chunking và so sánh ưu nhược điểm
- Xây dựng vector store với search, filter, và delete
- Kết nối knowledge base với agent qua RAG pattern
- Chỉ ra khi nào retrieval giúp ích và khi nào nó thất bại

---

## Cấu Trúc Lab: 2 Phase

### Phase 1 — Cá Nhân: Hoàn Thành src package

Mỗi sinh viên **tự mình** hoàn thành tất cả TODO trong `src/chunking.py`, `src/store.py`, và `src/agent.py`. `Document` dataclass và `FixedSizeChunker` đã được implement sẵn làm ví dụ.

### Phase 2 — Nhóm: So Sánh Retrieval Strategy

Nhóm cùng chọn một bộ tài liệu và thống nhất 5 benchmark queries. Mỗi thành viên **thử strategy riêng** (chunking, metadata), chạy cùng queries, rồi **so sánh kết quả trong nhóm** để học từ nhau.

---

## Thiết Lập Môi Trường

```bash
pip install -r requirements.txt
pytest tests/ -v          # Phần lớn tests sẽ FAIL (chưa implement)
```

Mặc định, lab vẫn chạy với `_mock_embed` nên **không bắt buộc** cài embedder thật.
File `.env` được tự động nạp khi chạy `main.py`. Với các Python snippet chạy trực tiếp, hãy `export` biến môi trường cần thiết hoặc gọi `load_dotenv()` nếu cần.

## Tùy Chọn Embedding Backend

### 1) Mặc định: Mock embedder

Không cần cài gì thêm ngoài:
```bash
pip install -r requirements.txt
```

### 2) Tùy chọn: Local embedder `all-MiniLM-L6-v2`

```bash
pip install sentence-transformers
python3 - <<'PY'
from src import LocalEmbedder
embedder = LocalEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

- Package `src` hỗ trợ `all-MiniLM-L6-v2` qua `sentence-transformers`.
- Lần chạy đầu tiên model sẽ được tải về và cache local.

### 3) Tùy chọn: OpenAI embedder

```bash
pip install openai
export OPENAI_API_KEY=your-key-here
python3 - <<'PY'
from src import OpenAIEmbedder
embedder = OpenAIEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

- Model mặc định cho lựa chọn này là `text-embedding-3-small`
- Có thể đổi model bằng:
```bash
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Quy tắc fallback

- Nếu không chọn gì, lab dùng `_mock_embed`
- Nếu chọn `local` hoặc `openai` nhưng setup thiếu, code sẽ tự fallback về `_mock_embed`
- Có thể cấu hình qua `.env` mà không cần `source .env`
- Script `main.py` chạy end-to-end và import public API từ package `src`

### Lệnh verify nhanh

Sau khi cài optional dependencies, có thể verify từng backend riêng:

**Verify local embedder**

```bash
python3 - <<'PY'
from src import LocalEmbedder

embedder = LocalEmbedder()
print(embedder._backend_name, len(embedder("embedding smoke test")))
PY
```

**Verify OpenAI embedder**

```bash
python3 - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
from src import OpenAIEmbedder

load_dotenv(dotenv_path=Path(".env"), override=False)
embedder = OpenAIEmbedder()
print(embedder._backend_name, len(embedder("embedding smoke test")))
PY
```

> Lưu ý: `OpenAIEmbedder` cần `OPENAI_API_KEY` hợp lệ trong môi trường hoặc `.env`.

---

## Cấu Trúc Thư Mục

```
├── README.md              ← Bạn đang đọc file này
├── exercises.md           ← Bài tập (4 phần)
├── main.py               ← Entry point cho manual demo
├── src/
│   ├── chunking.py       ← Chunking classes + similarity helper
│   ├── store.py          ← EmbeddingStore
│   ├── agent.py          ← KnowledgeBaseAgent
│   └── ...               ← Các module nhỏ hơn
├── data/                  ← Tài liệu mẫu + tài liệu nhóm (.txt/.md)
├── tests/
│   └── test_solution.py   ← Test suite (30+ tests)
├── report/
│   └── REPORT.md         ← Báo cáo (1 file/sinh viên)
├── docs/
│   ├── EVALUATION.md     ← Evaluation metrics
│   ├── INSTRUCTOR_GUIDE.md ← Instructor notes
│   └── SCORING.md        ← Tiêu chí chấm điểm
└── requirements.txt
```

---

## Các Giai Đoạn Lab

| Giai Đoạn | Hoạt Động |
|-----------|-----------|
| Chuẩn bị tài liệu | Nhóm chọn domain, thu thập tài liệu, chuyển sang .md/.txt |
| Lập trình cá nhân | Warm-up + implement tất cả TODO (cá nhân) |
| Thiết kế strategy | Mỗi người thử strategy riêng, thống nhất 5 queries |
| So sánh trong nhóm | Chạy benchmark, so sánh kết quả, chuẩn bị demo |
| Demo & thảo luận | Trình bày strategy + so sánh, thảo luận liên nhóm |

---

## Nhiệm Vụ Cá Nhân (Phase 1)

### Đã implement sẵn (tham khảo)
- `Document` dataclass — container cho text + metadata
- `FixedSizeChunker` — sliding window chunking

### Cần implement
- `SentenceChunker` — chia theo ranh giới câu
- `RecursiveChunker` — thử từng separator theo thứ tự
- `compute_similarity` — cosine similarity
- `ChunkingStrategyComparator` — so sánh 3 chiến lược
- `EmbeddingStore` — wrapper quanh vector store (5 methods)
- `KnowledgeBaseAgent` — RAG pattern agent

---

## Nhiệm Vụ Nhóm (Phase 2) — So Sánh Strategy

1. **Chọn bộ tài liệu** (5-10 docs): FAQ, SOP, policy, internal docs, hoặc domain bất kỳ
2. **Chuyển sang .txt/.md** nếu cần (xem tips trong exercises.md)
3. **Thống nhất 5 benchmark queries** kèm gold answers
4. **Mỗi thành viên thử strategy riêng**: chunking method, tham số, metadata schema
5. **So sánh kết quả trong nhóm**: strategy nào cho retrieval tốt hơn? Tại sao?

---

## Cách Tự Đánh Giá Kết Quả Retrieval

Khi chạy benchmark, đừng chỉ hỏi **"code có chạy không?"** mà hãy tự kiểm tra 5 góc nhìn sau:

1. **Retrieval Precision**
   - Top-3 có chứa chunk thật sự liên quan không?
   - Score có tách được kết quả tốt và nhiễu không?

2. **Chunk Coherence**
   - Chunk có giữ được ý trọn vẹn không?
   - Strategy nào làm chunk dễ đọc và dễ retrieve hơn?

3. **Metadata Utility**
   - `search_with_filter()` có giúp tăng độ chính xác không?
   - Filter có quá chặt, làm mất kết quả tốt không?

4. **Grounding Quality**
   - Câu trả lời của agent có thật sự dựa trên retrieved context không?
   - Có thể chỉ ra chunk nào hỗ trợ câu trả lời không?

5. **Data Strategy Impact**
   - Bộ tài liệu nhóm chọn có phù hợp với benchmark queries không?
   - Strategy chunking / metadata của bạn có hợp với domain không?

> Xem `docs/EVALUATION.md` nếu bạn muốn một checklist chi tiết hơn cho phần này.

---

## Chấm Điểm

Xem chi tiết tại `docs/SCORING.md`. Tóm tắt:

| Phần | Điểm |
|------|------|
| Cá nhân (code + phân tích) | 60 |
| Nhóm (strategy + so sánh) | 40 |
| **Tổng** | **100** |

---

## Sản Phẩm Nộp Bài

1. `src/` — hoàn thành tất cả TODO cần thiết
2. `report/REPORT.md` — một báo cáo/sinh viên (gồm cả phần nhóm và cá nhân)

---

## Chạy Kiểm Thử

```bash
pytest tests/ -v
```
