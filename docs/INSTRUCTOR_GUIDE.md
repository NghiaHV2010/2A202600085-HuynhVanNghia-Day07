# Instructor Guide: Lab 7 - Data Foundations: Embedding & Vector Store

Hướng dẫn này dành cho giảng viên để dẫn dắt buổi lab 4.5 giờ. Lab chia làm 2 pha: **cá nhân** (code) và **nhóm** (so sánh strategy, học từ nhau).

---

## Mục Tiêu Học Tập Cốt Lõi

1. **Embedding Intuition (G2)**: Hiểu cosine similarity, dự đoán được điểm tương đồng, nhận ra giới hạn của embedding.
2. **Vector Store Operations (G3)**: Triển khai store/search/filter/delete; giải thích khi nào metadata filtering giúp ích vs gây hại.
3. **Full Pipeline (G4)**: Triển khai mỗi bước Document → Chunk → Embed → Store → Query → Inject; so sánh chunking strategies.
4. **Data Strategy (G5)**: Chọn dữ liệu, thiết kế metadata, tối ưu chunking — hiểu rằng data quality > model selection.

---

## Ghi Chú Cho Giảng Viên: Embedder Thật Là Tùy Chọn

- Lab này **không bắt buộc** sinh viên cài embedder thật.
- Luồng mặc định cho lớp học vẫn là `_mock_embed`, nên sinh viên vẫn có thể hoàn thành lab và pass test mà không cần tải model nào.
- Nếu sinh viên muốn thử embedding thật trên máy cá nhân, package `src` đã hỗ trợ cả:
  - `all-MiniLM-L6-v2` qua `sentence-transformers`
  - OpenAI embeddings qua package `openai`

Ví dụ local embedder:

```bash
pip install sentence-transformers
python3 - <<'PY'
from src import LocalEmbedder
embedder = LocalEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

Ví dụ OpenAI embedder:

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

- Khuyến nghị giảng viên nói rõ ngay từ đầu: **“Local/OpenAI embedder là bonus / optional, không phải điều kiện để hoàn thành lab.”**
- Khi có sinh viên máy yếu, mạng chậm, không có API key, hoặc không muốn tải model, hãy hướng họ tiếp tục với `_mock_embed` để tránh bị kẹt ở phần setup.
- `src_w_solution/` là reference solution cho giảng viên / maintainer. Không phân phối thư mục này cho sinh viên.

---

## Timeline & Flow (4.5 giờ)

### Phase 1: Document Preparation (30 phút, 0:00–0:30)

**Hoạt động (nhóm):**
- Nhóm chọn domain (FAQ, law, recipes, medical, tech docs, v.v.)
- Thu thập 5-10 tài liệu, chuyển sang `.txt`/`.md`, đặt vào `data/`
- Thiết kế metadata schema (ít nhất 2 trường hữu ích)

**Vai trò giảng viên:**
- Giải thích lab structure: "30 phút chuẩn bị tài liệu nhóm → mỗi người tự code → mỗi người thử strategy riêng → so sánh trong nhóm → demo với lớp"
- Gợi ý domain nếu nhóm chưa quyết
- Nhấn mạnh: "Chọn tài liệu có cấu trúc rõ ràng — chất lượng tài liệu quyết định kết quả"

### Phase 2: Individual Coding (90 phút, 0:30–2:00)

**Warm-up (10 phút):**
- Ex 1.1: Cosine similarity — giải thích bằng ngôn ngữ tự nhiên
- Ex 1.2: Chunking math — tính toán số chunks

**Implementation (80 phút):**
- Mỗi sinh viên **tự mình** implement tất cả TODO trong `src/chunking.py`, `src/store.py`, và `src/agent.py`
- `Document` và `FixedSizeChunker` đã implement sẵn làm ví dụ
- Thứ tự gợi ý: `SentenceChunker` → `RecursiveChunker` → `compute_similarity` → `ChunkingStrategyComparator` → `EmbeddingStore` → `KnowledgeBaseAgent`

**Vai trò giảng viên:**
- **Nhấn mạnh**: "Đây là phần cá nhân — mỗi người tự code"
- **Checkpoint 1 (1:00)**: "Ai đã pass phần chunking (`TestSentenceChunker`, `TestRecursiveChunker`)?" — giải thích nếu < 50%
- **Checkpoint 2 (1:30)**: "Ai đã pass TestEmbeddingStore?" — debug nếu cần

### Phase 3: Strategy Design (60 phút, 2:00–3:00)

**Hoạt động:**
- Nhóm thống nhất **5 benchmark queries + gold answers**
- Mỗi thành viên **chọn strategy riêng** (chunking method, tham số, metadata schema)
- Chạy baseline comparison, thiết kế custom strategy nếu muốn
- Index tài liệu vào EmbeddingStore với strategy riêng

**Vai trò giảng viên:**
- Khuyến khích mỗi người thử strategy khác nhau: "Một người thử `FixedSizeChunker`, một người thử `RecursiveChunker`, một người thử custom"
- Kiểm tra benchmark queries: "Queries có đủ đa dạng không?"
- Nhắc: gold answers phải cụ thể, verifiable

**Checkpoint (2:45):** Mỗi nhóm phải có 5 queries + gold answers sẵn sàng

### Phase 4: So Sánh & Thảo Luận Trong Nhóm (30 phút, 3:00–3:30)

**Hoạt động:**
1. Mỗi thành viên chạy 5 benchmark queries với strategy riêng (10 phút)
2. So sánh kết quả trong nhóm (10 phút):
   - Strategy nào tốt nhất? Tại sao?
   - Có query nào strategy A thắng nhưng B thua?
3. Chuẩn bị demo (10 phút): chọn insights hay nhất để chia sẻ

**Vai trò giảng viên:**
- Đi quanh lớp, hỏi: "Strategy nào thắng? Giải thích được tại sao không?"
- Thu thập 2-3 insights hay từ các nhóm để dùng trong phần demo discussion

### Phase 5: Demo & Discussion Liên Nhóm (60 phút, 3:30–4:30)

**Format demo (8-10 phút/nhóm):**
1. Giới thiệu domain + document set (1 phút)
2. Mỗi thành viên tóm tắt strategy của mình (2 phút)
3. So sánh: strategy nào thắng trên data này? Tại sao? (3 phút)
4. Demo 1-2 queries live (2 phút)
5. Q&A từ nhóm khác + giảng viên (2 phút)

**Câu hỏi gợi ý cho discussion:**
- "Nếu chuyển sang domain khác, strategy nào vẫn hoạt động tốt?"
- "Metadata filtering giúp ích ở đâu? Ở đâu nó làm mất kết quả tốt?"
- "Từ kết quả nhóm bạn, nhóm mình có thể áp dụng gì?"

**Wrap-up giảng viên (5 phút):**
- Key lesson: "Cùng tài liệu, khác strategy → kết quả rất khác. Hiểu tại sao quan trọng hơn chạy được."
- Nhắc: mỗi sinh viên nộp 1 report (phần nhóm giống nhau, phần cá nhân + strategy khác nhau)
- Kết nối với Day 8 (RAG pipeline hoàn chỉnh)

---

## Sai Lầm Phổ Biến

| Sai lầm | Cách xử lý |
|---------|------------|
| **Overlap > chunk_size** | Hỏi: "step = chunk_size - overlap. Nếu overlap >= chunk_size thì step là gì?" |
| **Quên normalize vector** trong compute_similarity | Chỉ ra công thức: cần chia cho \|\|a\|\| * \|\|b\|\| |
| **search_with_filter không lọc trước** | Sinh viên search rồi mới filter → kết quả sai. Phải filter trước, rồi search |
| **KnowledgeBaseAgent không inject context** | Kiểm tra: prompt có chứa retrieved chunks không? |
| **Tất cả thành viên chọn cùng strategy** | Yêu cầu mỗi người thử strategy khác — mục tiêu là so sánh |
| **Benchmark queries quá dễ/giống nhau** | Yêu cầu đa dạng: factual, multi-chunk, metadata-dependent |

---

## Tiêu Chí Thành Công

Buổi lab thành công nếu:
- Mọi sinh viên pass được ít nhất 70% tests (cá nhân)
- Mỗi nhóm có ít nhất 2 strategies khác nhau để so sánh
- Sinh viên giải thích được tại sao strategy A tốt hơn B trên data cụ thể
- Demo có discussion sôi nổi giữa các nhóm
- Sinh viên kết nối được: data strategy ảnh hưởng trực tiếp đến retrieval quality

---

*"Data quality thường quan trọng hơn đổi sang model đắt hơn. Dạy sinh viên nhìn vào data trước khi nhìn vào model."*
