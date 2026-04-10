METADATA FILTER EVALUATION: search() vs search_with_filter()

────────────────────────────────────────────────────────────────────────────────────────────────────
  Q1: Bệnh Alzheimer có di truyền không?
  Expected category: Thần kinh
  Auto-detected filter: Thần kinh

  Rank  search() — no filter                            search_with_filter()
  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────
  [1]   x Tiêu hóa - Hậu môn trự score=0.3665           v Thần kinh              score=0.2454
  [2]   x Hô hấp                 score=0.3420           v Thần kinh              score=0.2092
  [3]   x Tiêu hóa - Gan mật     score=0.2949           v Thần kinh              score=0.2054

  Precision (no filter): 0%   |   Precision (filtered): 100%   |   Overlap: 0/3

────────────────────────────────────────────────────────────────────────────────────────────────────
  Q2: Nguyên nhân gây ăn không tiêu là gì?
  Expected category: Tiêu hóa - Gan mật
  Auto-detected filter: Tiêu hóa - Gan mật

  Rank  search() — no filter                            search_with_filter()
  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────
  [1]   x Hô hấp                 score=0.3630           v Tiêu hóa - Gan mật     score=0.2114
  [2]   x Thần kinh              score=0.3495           v Tiêu hóa - Gan mật     score=0.1567
  [3]   x Nội tiết - Đái tháo đư score=0.3227           v Tiêu hóa - Gan mật     score=0.1182

  Precision (no filter): 0%   |   Precision (filtered): 100%   |   Overlap: 0/3

────────────────────────────────────────────────────────────────────────────────────────────────────
  Q3: Áp xe hậu môn có tự khỏi không?
  Expected category: Tiêu hóa - Hậu môn trực tràng
  Auto-detected filter: Tiêu hóa - Hậu môn trực tràng

  Rank  search() — no filter                            search_with_filter()
  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────
  [1]   x Hô hấp                 score=0.3209           v Tiêu hóa - Hậu môn trự score=0.2589
  [2]   x Thần kinh              score=0.3062           v Tiêu hóa - Hậu môn trự score=0.2100
  [3]   v Tiêu hóa - Hậu môn trự score=0.2589           v Tiêu hóa - Hậu môn trự score=0.1069

  Precision (no filter): 33%   |   Precision (filtered): 100%   |   Overlap: 1/3

────────────────────────────────────────────────────────────────────────────────────────────────────
  Q4: Triệu chứng của áp xe phổi?
  Expected category: Hô hấp
  Auto-detected filter: Hô hấp

  Rank  search() — no filter                            search_with_filter()
  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────
  [1]   x Thần kinh              score=0.3118           v Hô hấp                 score=0.1453
  [2]   x Sản phụ khoa           score=0.3053           v Hô hấp                 score=0.1193
  [3]   x Thần kinh              score=0.3033           v Hô hấp                 score=0.1024

  Precision (no filter): 0%   |   Precision (filtered): 100%   |   Overlap: 0/3

────────────────────────────────────────────────────────────────────────────────────────────────────
  Q5: Bàn chân đái tháo đường chăm sóc thế nào?
  Expected category: Nội tiết - Đái tháo đường
  Auto-detected filter: Nội tiết - Đái tháo đường

  Rank  search() — no filter                            search_with_filter()
  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────
  [1]   x Tiết niệu              score=0.3342           v Nội tiết - Đái tháo đư score=0.2847
  [2]   x Tiêu hóa - Gan mật     score=0.3076           v Nội tiết - Đái tháo đư score=0.1981
  [3]   x Thần kinh              score=0.2881           v Nội tiết - Đái tháo đư score=0.1557

  Precision (no filter): 0%   |   Precision (filtered): 100%   |   Overlap: 0/3

────────────────────────────────────────────────────────────────────────────────────────────────────
  Q6: Băng huyết sau sinh xử lý ra sao?
  Expected category: Sản phụ khoa
  Auto-detected filter: Sản phụ khoa

  Rank  search() — no filter                            search_with_filter()
  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────
  [1]   v Sản phụ khoa           score=0.2719           v Sản phụ khoa           score=0.2719
  [2]   x Thần kinh              score=0.2684           v Sản phụ khoa           score=0.0911
  [3]   x Nội tiết - Đái tháo đư score=0.2570           v Sản phụ khoa           score=0.0877

  Precision (no filter): 33%   |   Precision (filtered): 100%   |   Overlap: 1/3

────────────────────────────────────────────────────────────────────────────────────────────────────
  Q7: Bàng quang tăng hoạt là gì?
  Expected category: Tiết niệu
  Auto-detected filter: Tiết niệu

  Rank  search() — no filter                            search_with_filter()
  ───── ─────────────────────────────────────────────── ───────────────────────────────────────────────
  [1]   x Thần kinh              score=0.3066           v Tiết niệu              score=0.1974
  [2]   x Nội tiết - Đái tháo đư score=0.2571           v Tiết niệu              score=0.1601
  [3]   x Thần kinh              score=0.2457           v Tiết niệu              score=0.1372

  Precision (no filter): 0%   |   Precision (filtered): 100%   |   Overlap: 0/3

  [3]   x Thần kinh              score=0.2457           v Tiết niệu              score=0.1372

  Precision (no filter): 0%   |   Precision (filtered): 100%   |   Overlap: 0/3 

────────────────────────────────────────────────────────────────────────────────────────────────────
  Total queries evaluated:         7
  Avg precision (no filter):       9.5%
  Avg precision (no filter):       9.5%
  Avg precision (with filter):     100.0%
  Precision improvement:           +90.5%
Đang hiển thị 712548391865381266.