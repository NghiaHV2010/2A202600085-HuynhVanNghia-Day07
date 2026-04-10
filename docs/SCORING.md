# Lab Scoring Rubric: Embedding & Vector Store

Mỗi sinh viên nộp **một báo cáo duy nhất** (`report/REPORT.md`) bao gồm cả phần cá nhân và phần nhóm.

> Tham khảo thêm `docs/EVALUATION.md` để xem các metric và góc nhìn đánh giá retrieval quality.

---

## Điểm Cá Nhân (60 Điểm)

| Hạng mục | Mô tả | Điểm |
| :--- | :--- | :--- |
| **Core Implementation** | Tất cả pytest tests pass (`pytest tests/ -v`) | 30 |
| **My Approach** | Giải thích cách implement từng phần trong package src | 10 |
| **Competition Results** | 5 benchmark queries chạy trên package src cá nhân, cùng bộ queries với nhóm | 10 |
| **Warm-up** | Cosine similarity explanation + chunking math | 5 |
| **Similarity Predictions** | 5 cặp câu, dự đoán vs actual, reflection | 5 |

---

## Điểm Nhóm (40 Điểm)

| Hạng mục | Mô tả | Điểm |
| :--- | :--- | :--- |
| **Strategy Design** | Giải thích strategy cá nhân + rationale + so sánh với baseline và với thành viên khác | 15 |
| **Document Set Quality** | 5-10 tài liệu có chủ đề rõ ràng, metadata hữu ích, nguồn minh bạch | 10 |
| **Retrieval Quality** | Precision trên 5 benchmark queries (top-3 có relevant chunks) | 10 |
| **Demo** | Trình bày strategy, so sánh trong nhóm, bài học rút ra | 5 |

### Cách Tính Retrieval Quality (10 điểm)

Nhóm thống nhất **5 benchmark queries** kèm **gold answers**. Mỗi thành viên chạy queries trên strategy riêng.

**Chấm mỗi query (2 điểm/query):**
- 2 điểm: Top-3 chứa chunk relevant + agent answer chính xác
- 1 điểm: Top-3 có relevant chunk nhưng answer thiếu chi tiết hoặc relevant chunk không ở top-1
- 0 điểm: Không retrieve được chunk relevant trong top-3

---

## Tính Điểm Tổng

**Tổng = Cá Nhân (60) + Nhóm (40) = 100 Điểm Tối Đa**

> [!IMPORTANT]
> **Một báo cáo, hai góc nhìn**: Phần nhóm (document selection, strategy) sẽ giống nhau giữa các thành viên. Phần cá nhân (approach, results, reflection) phải khác nhau vì mỗi người code riêng.

> [!IMPORTANT]
> **Strategy > Performance**: 15 điểm cho strategy design vs 10 điểm cho retrieval quality. Chúng tôi đánh giá cao khả năng **suy nghĩ và giải thích** hơn là điểm số thuần tuý.

> [!IMPORTANT]
> **Học từ nhau**: Mỗi thành viên thử strategy riêng trên cùng bộ tài liệu và cùng queries. So sánh kết quả trong nhóm giúp hiểu tại sao strategy A tốt hơn B. Phần demo là cơ hội thảo luận với các nhóm khác về document selection, strategy, và kết quả.
