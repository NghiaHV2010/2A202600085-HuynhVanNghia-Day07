# Evaluation Metrics for Lab 7: Embedding & Vector Store

Trong lab này, chúng ta không chỉ hỏi "Nó chạy không?" mà hỏi **"Retrieval quality tốt đến đâu?"**.

> Xem `docs/SCORING.md` để biết rubric chấm điểm chính thức.

## Các Metric Quan Trọng

### 1. Retrieval Precision

- **Top-k Relevance**: Trong k kết quả trả về, bao nhiêu kết quả thực sự liên quan đến câu hỏi?
- **Score Distribution**: Điểm similarity có phân biệt rõ giữa kết quả tốt và kết quả nhiễu không?
- **Benchmark Score**: Mỗi query 2 điểm — top-3 relevant + answer chính xác = 2, thiếu = 1/0.
- **Mục tiêu**: Top-3 results nên có ít nhất 2 kết quả liên quan trực tiếp.

### 2. Chunk Coherence

- **Semantic Completeness**: Chunk có giữ nguyên ý hoàn chỉnh hay bị cắt giữa câu/giữa ý?
- **Context Preservation**: Overlap có giúp giữ liên kết giữa các chunks không?
- **Đo lường**: So sánh chunk count và avg_length giữa 3 chiến lược, đánh giá chủ quan mức độ coherent.

### 3. Metadata Utility

- **Filter Effectiveness**: Khi lọc theo metadata (category, lang, date), kết quả có chính xác hơn không?
- **Recall Trade-off**: Filter quá chặt có làm mất kết quả tốt không?
- **Đo lường**: So sánh top-3 results giữa `search()` và `search_with_filter()` trên cùng query.

### 4. Grounding Quality

- **Factual Accuracy**: Câu trả lời của KnowledgeBaseAgent có dựa trên context retrieved hay bịa?
- **Source Traceability**: Có thể chỉ ra chunk nào đã được dùng để trả lời không?
- **Đo lường**: Verify agent answer against gold answer trong benchmark.

### 5. Data Strategy Impact

- **Document Selection**: Tài liệu có chủ đề rõ ràng, đủ nội dung để retrieval có ý nghĩa?
- **Metadata Design**: Schema metadata có giúp lọc kết quả tốt hơn?
- **Chunking Rationale**: Strategy chunking có khai thác cấu trúc domain?
- **Đo lường**: So sánh retrieval score giữa các thành viên trong nhóm — cùng tài liệu, khác strategy.

## Cách Sử Dụng Metrics

Các metric trên không yêu cầu tính toán phức tạp. Sinh viên nên:

1. **Quan sát có hệ thống**: Chạy cùng query với các cấu hình khác nhau, ghi lại kết quả
2. **So sánh A/B**: filtered vs unfiltered, strategy A vs strategy B, data set X vs data set Y
3. **Giải thích tại sao**: Không chỉ ghi kết quả, mà giải thích nguyên nhân

Kết quả đánh giá nên được ghi trong individual report (cá nhân) và group report (nhóm).
