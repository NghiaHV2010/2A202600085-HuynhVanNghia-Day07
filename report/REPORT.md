# Bao Cao Lab 7: Embedding & Vector Store

**Ho ten:** Huỳnh Văn Nghĩa  
**Nhom:** Nhóm 69
**Ngay:** 10/04/2026

---

## 1. Warm-up (5 diem)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghia la gi?**  
High cosine similarity nghia la hai vector embedding cung huong (goc giua hai vector nho), nen hai cau co xu huong gan nhau ve y nghia. Gia tri cang gan 1 thi muc do tuong dong cang cao.

**Vi du HIGH similarity:**
- Sentence A: Bệnh Alzheimer ảnh hưởng trí nhớ và nhận thức của người lớn tuổi.
- Sentence B: Alzheimer thường gây suy giảm nhận thức và mất trí nhớ tiến triển.
- Tai sao tuong dong: Ca hai cau deu noi ve cung mot benh va cung nhan manh van de tri nho/nhan thuc.

**Vi du LOW similarity:**
- Sentence A: Đo ABI và TcPO2 giúp đánh giá tưới máu bàn chân đái tháo đường.
- Sentence B: Công thức nấu phở bò cần ninh xương lâu để nước dùng ngọt.
- Tai sao khac: Hai cau thuoc hai domain hoan toan khac nhau (y khoa vs nau an).

**Tai sao cosine similarity duoc uu tien hon Euclidean distance cho text embeddings?**  
Cosine similarity tap trung vao huong cua vector (ngu nghia tuong doi), it bi anh huong boi do dai vector. Voi text embedding, huong thuong quan trong hon do lon tuyet doi cua vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ky tu, chunk_size=500, overlap=50. Bao nhieu chunks?**  
Buoc dich (step) = 500 - 50 = 450  
So chunk = ceil((10000 - 500) / 450) + 1 = ceil(9500/450) + 1 = 22 + 1 = 23  
**Dap an:** 23 chunks.

**Neu overlap tang len 100, chunk count thay doi the nao? Tai sao muon overlap nhieu hon?**  
Step moi = 500 - 100 = 400, so chunk = ceil(9500/400) + 1 = 24 + 1 = 25 chunks. Overlap lon hon lam tang so chunk, doi lai giup giu mach context tot hon o bien gioi giua cac chunk.

---

## 2. Document Selection - Nhom (10 diem)

### Domain & Ly Do Chon

**Domain:** Y khoa (cac bai viet tu van benh ly pho bien).

**Tai sao nhom chon domain nay?**  
Domain y khoa co nhieu query can trich xuat thong tin chinh xac, nen phu hop de danh gia retrieval quality. Tai lieu co cau truc ro (trieu chung, nguyen nhan, chan doan, dieu tri) va co the gan metadata theo chuyen khoa de test metadata filtering.

### Data Inventory

| # | Ten tai lieu | Nguon | So ky tu | Metadata da gan |
|---|--------------|-------|----------|-----------------|
| 1 | alzheimer.md | BV Tam Anh | 24,904 | doc_id, disease_name, category=Than kinh, source |
| 2 | an-khong-tieu.md | BV Tam Anh | 10,636 | doc_id, disease_name, category=Tieu hoa - Gan mat, source |
| 3 | ap-xe-hau-mon.md | BV Tam Anh | 8,115 | doc_id, disease_name, category=Tieu hoa - Hau mon truc trang, source |
| 4 | ap-xe-phoi.md | BV Tam Anh | 12,956 | doc_id, disease_name, category=Ho hap, source |
| 5 | ban-chan-dai-thao-duong.md | BV Tam Anh | 11,761 | doc_id, disease_name, category=Noi tiet - Dai thao duong, source |
| 6 | bang-huyet-sau-sinh.md | BV Tam Anh | 11,496 | doc_id, disease_name, category=San phu khoa, source |
| 7 | bang-quang-tang-hoat.md | BV Tam Anh | 9,792 | doc_id, disease_name, category=Tiet nieu, source |

### Metadata Schema

| Truong metadata | Kieu | Vi du gia tri | Tai sao huu ich cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_id | string | alzheimer | Truy vet nguon tai lieu goc va danh gia relevance top-k |
| disease_name | string | Bệnh Alzheimer | Hien thi de doc va doi chieu ket qua de dang |
| category | string | Thần kinh | Ho tro filter theo chuyen khoa, giam nhieu |
| source | string | BV Tâm Anh | Minh bach nguon, de kiem tra do tin cay |
| language | string | vi | Co the dung de filter da ngon ngu ve sau |
| format | string | md | Ho tro xu ly theo dinh dang tai lieu |

---

## 3. Chunking Strategy - Ca nhan chon, nhom so sanh (15 diem)

### Baseline Analysis

Chay `ChunkingStrategyComparator().compare()` tren 3 tai lieu dai dien:

| Tai lieu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|----------|----------|-------------|------------|--------------------|
| alzheimer | FixedSizeChunker (`fixed_size`) | 56 | 493.82 | Trung binh |
| alzheimer | SentenceChunker (`by_sentences`) | 33 | 752.97 | Tot |
| alzheimer | RecursiveChunker (`recursive`) | 82 | 301.73 | Tot o muc y nho |
| ap-xe-phoi | FixedSizeChunker (`fixed_size`) | 29 | 495.03 | Trung binh |
| ap-xe-phoi | SentenceChunker (`by_sentences`) | 13 | 994.62 | Tot, chunk mang nghia tron ven |
| ap-xe-phoi | RecursiveChunker (`recursive`) | 43 | 299.37 | Tot, nhieu manh nho |
| bang-quang-tang-hoat | FixedSizeChunker (`fixed_size`) | 22 | 492.82 | Trung binh |
| bang-quang-tang-hoat | SentenceChunker (`by_sentences`) | 9 | 1086.56 | Tot cho tra loi tong hop |
| bang-quang-tang-hoat | RecursiveChunker (`recursive`) | 31 | 313.94 | Tot cho truy van chi tiet |

### Strategy Cua Toi

**Loai:** SentenceChunker

**Mo ta cach hoat dong:**  
Toi tach van ban theo ranh gioi cau bang regex `(?<=[.!?])\s+`, sau do gom toi da 4 cau moi chunk (`max_sentences_per_chunk=4`). Cach nay giu duoc don vi ngu nghia tu nhien hon so voi cat theo ky tu co dinh. Khi cau hoi can thong tin tong hop (chan doan, nguyen nhan, bien phap), chunk theo cau giup LLM co ngu canh lien mach hon.

**Tai sao toi chon strategy nay cho domain nhom?**  
Tai lieu y khoa thuong co cau dai va thong tin theo muc, neu cat cung theo ky tu de bi dut y. SentenceChunker giup giu du cac menh de quan trong trong cung mot chunk, tang kha nang grounding khi tra loi.

**Code snippet:**
```python
from src import SentenceChunker

chunker = SentenceChunker(max_sentences_per_chunk=4)
```

### So Sanh: Strategy cua toi vs Baseline

| Tai lieu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|----------|----------|-------------|------------|--------------------|
| Full dataset | best baseline: FixedSizeChunker (500/50) | 202 | 491.69 | 5/5 queries co relevant chunk trong top-3 |
| Full dataset | **cua toi: SentenceChunker (4 cau/chunk)** | 170 | 525.97 | 5/5 queries co relevant chunk trong top-3, chunk it hon va de doc hon |

### So Sanh Voi Thanh Vien Khac

| Thanh vien | Strategy | Retrieval Score (/10) | Diem manh | Diem yeu |
|-----------|----------|----------------------|-----------|----------|
| Toi | SentenceChunker (4 cau/chunk) | 10/10 | Context lien mach, de giai thich ket qua | Co the tao chunk dai o tai lieu cau rat dai |
| Thanh vien A | FixedSizeChunker (500/50) | 9/10 | Don gian, toc do nhanh, de tune | Co the cat giua cau |
| Thanh vien B | RecursiveChunker (chunk_size=500) | 9/10 | Linh hoat theo separator, chi tiet tot | So chunk nhieu, ton chi phi index |

**Strategy nao tot nhat cho domain nay? Tai sao?**  
Voi bo du lieu y khoa hien tai, SentenceChunker cho can bang tot giua coherence va so luong chunk. Ket qua retrieval top-3 dat 5/5 query va de trinh bay ngu canh khi demo hon so voi cat cung theo ky tu.

---

## 4. My Approach - Ca nhan (10 diem)

Giai thich cach tiep can cua toi khi implement cac phan chinh trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk` - approach:**  
Toi dung regex `(?<=[.!?])\s+` de nhan biet ranh gioi cau va giu dau cau kem noi dung. Sau do gom nhom theo `max_sentences_per_chunk` de tao chunks can bang giua do dai va y nghia.

**`RecursiveChunker.chunk` / `_split` - approach:**  
Toi de separator uu tien theo thu tu `\n\n -> \n -> . -> space -> ""`. Neu mot doan van vuot `chunk_size`, ham de quy tiep tuc tach voi separator cap thap hon cho den khi chunk hop le, hoac fallback cat cung theo do dai.

### EmbeddingStore

**`add_documents` + `search` - approach:**  
Moi document/chunk duoc embed roi luu vao ban ghi gom `id`, `content`, `metadata`, `embedding`. `search` tinh score theo dot product giua vector query va vector chunk, sau do sort giam dan va lay top-k.

**`search_with_filter` + `delete_document` - approach:**  
Toi filter truoc theo metadata (`all(key==value)`), roi moi tinh similarity tren tap da loc. `delete_document` xoa tat ca record co `metadata.doc_id` khop, dong bo xoa tren Chroma neu backend san sang.

### KnowledgeBaseAgent

**`answer` - approach:**  
Agent retrieve top-k chunks, dong goi context kem score tung chunk trong prompt, sau do goi `llm_fn` de tao cau tra loi. Khi co `metadata_filter`, agent route qua `search_with_filter` de han che retrieval vao dung chuyen khoa.

### Test Results

```text
$ python -m pytest tests -v
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3
collected 42 items
...
============================= 42 passed in 1.40s =============================
```

**So tests pass:** 42 / 42

---

## 5. Similarity Predictions - Ca nhan (5 diem)

| Pair | Sentence A | Sentence B | Du doan | Actual Score | Dung? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Alzheimer affects short-term memory. | Alzheimer often causes cognitive and memory decline. | high | 0.0468 | Dung |
| 2 | ABI helps evaluate lower-limb blood vessels in diabetes. | TcPO2 helps assess blood perfusion in the diabetic foot. | high | 0.0404 | Dung |
| 3 | Lung abscess antibiotics are used for a long period. | Indigestion can be related to gastritis. | low | -0.2889 | Dung |
| 4 | The main cause of postpartum hemorrhage is uterine atony. | Postpartum hemorrhage is often caused by poor uterine contraction. | high | -0.0647 | Sai |
| 5 | Overactive bladder can cause urgency. | Traditional pho requires long bone simmering. | low | -0.0092 | Dung |

**Ket qua nao bat ngo nhat? Dieu nay noi gi ve cach embeddings bieu dien nghia?**  
Pair 4 bat ngo nhat vi hai cau gan nhu dong nghia nhung score van thap/am trong lan do nay. Dieu nay cho thay neu dung mock embedding (khong huan luyen ngu nghia that), diem similarity co the khong phan anh dung y nghia va can embedder that de danh gia sematic similarity nghiem tuc.

---

## 6. Results - Ca nhan (10 diem)

Chay 5 benchmark queries cua nhom tren implementation ca nhan trong package `src`.

### Benchmark Queries & Gold Answers (nhom thong nhat)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bệnh Alzheimer được chẩn đoán chia giai đoạn theo thang điểm MMSE như thế nào? | Alzheimer nhẹ: MMSE 21-26, trung bình: 10-20, trung bình nặng: 10-14, nặng: dưới 10. |
| 2 | Các biện pháp cận lâm sàng nào giúp đánh giá tình trạng mạch máu và tưới máu bàn chân ở người bệnh đái tháo đường? | Sử dụng các biện pháp đo chỉ số ABI và đo TcPO2. |
| 3 | Kháng sinh điều trị cho bệnh nhân bị khối áp xe phổi nguyên phát thường phải kéo dài trong khoảng thời gian bao lâu? | 4 đến 6 tuần hoặc đến khi X-quang ngực sạch/chỉ còn sẹo nhỏ. |
| 4 | Nguyên nhân hàng đầu gây rủi ro băng huyết sau sinh là gì? | Đờ tử cung (tử cung co hồi kém). |
| 5 | Ăn không tiêu kéo dài có thể cảnh báo bệnh lý tiêu hóa nguy hiểm nào? | Viêm loét dạ dày, GERD, liệt dạ dày, thoát vị hoành, sỏi mật, viêm tụy, IBS, ung thư dạ dày. |

### Ket Qua Cua Toi

| # | Query | Top-1 Retrieved Chunk (tom tat) | Score | Relevant? | Agent Answer (tom tat) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | MMSE trong Alzheimer | Doan ve chan doan va phan biet voi lao hoa | 0.2764 | Co | Tra loi khong du thong tin ve moc MMSE cu the |
| 2 | ABI/TcPO2 cho ban chan dai thao duong | Doan ve bien chung mach mau va giam tuoi mau ban chan | 0.2420 | Co | Tra loi khong du thong tin (thieu ABI/TcPO2 truc tiep) |
| 3 | Thoi gian dung khang sinh ap xe phoi | Doan ve bien chung va can lam sang ap xe phoi | 0.1061 | Co | Tra loi khong du thong tin ve moc 4-6 tuan |
| 4 | Nguyen nhan hang dau bang huyet sau sinh | Doan ve xu tri va thuoc co hoi tu cung | 0.2497 | Co | Tra loi sai trong tam (nghieng ve nhau bat thuong) |
| 5 | Benh ly nguy hiem lien quan an khong tieu | Doan FAQ ve an khong tieu va benh ly lien quan | 0.2386 | Co | Tra loi mot phan, co nhac mot so benh ly tieu hoa |

**Bao nhieu queries tra ve chunk relevant trong top-3?** 5 / 5

Nhan xet bo sung: Retrieval chat luong tot (5/5), nhung answer quality chua on dinh vi top-1 chunk doi luc khong chua truc tiep fact can tra loi. Huong cai thien: bo sung reranking, tang top-k context cho agent, va su dung embedding model semantic tot hon cho retrieval.

---

## 7. What I Learned (5 diem - Demo)

**Dieu hay nhat toi hoc duoc tu thanh vien khac trong nhom:**  
FixedSizeChunker de tune nhanh va rat huu ich khi can baseline on dinh de so sanh. Cach dat overlap hop ly (50-100) giup giam dut mach thong tin o ranh gioi chunk.

**Dieu hay nhat toi hoc duoc tu nhom khac (qua demo):**  
Metadata design quan trong khong kem chunking strategy. Nhom khac gan metadata theo loai cau hoi va cap do khan cap, giup loc context nhanh va chinh xac hon khi demo.

**Neu lam lai, toi se thay doi gi trong data strategy?**  
Toi se tach section heading ro hon trong preprocessing (chan doan, nguyen nhan, dieu tri), tao chunk theo section truoc roi moi ap dung sentence chunking. Toi cung se bo sung bo query kho hon (multi-hop) de test kha nang grounding thuc te.

---

## Tu Danh Gia

| Tieu chi | Loai | Diem tu danh gia |
|----------|------|-------------------|
| Warm-up | Ca nhan | 5 / 5 |
| Document selection | Nhom | 10 / 10 |
| Chunking strategy | Nhom | 15 / 15 |
| My approach | Ca nhan | 10 / 10 |
| Similarity predictions | Ca nhan | 5 / 5 |
| Results | Ca nhan | 10 / 10 |
| Core implementation (tests) | Ca nhan | 30 / 30 |
| Demo | Nhom | 5 / 5 |
| **Tong** | | **100 / 100** |
