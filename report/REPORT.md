# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Quách Ngọc Quang
**Nhóm:** C401-E6
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nó thể hiện mức độ tương đồng về hướng của hai vector trong không gian đa chiều, tức là hai đoạn văn bản có ý nghĩa ngữ nghĩa (semantic meaning) rất gần nhau, bất kể độ dài ngắn khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Học máy là một tập con của trí tuệ nhân tạo."
- Sentence B: "Machine learning là một nhánh thuộc lĩnh vực AI."
- Tại sao tương đồng: Cả hai đều cùng nói về một mối quan hệ phân cấp giữa AI và Học máy, dù dùng từ ngữ khác nhau.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi đang viết mã Python."
- Sentence B: "Ngày mai trời có thể sẽ mưa."
- Tại sao khác: Nội dung hoàn toàn khác biệt, không có sự liên quan về chủ đề hay ngữ cảnh.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì Cosine Similarity không bị ảnh hưởng bởi độ dài văn bản (magnitude-invariant). Một tài liệu dài và một bản tóm tắt ngắn của nó sẽ có độ dài vector rất khác nhau (Euclidean lớn), nhưng hướng của chúng lại rất giống nhau (Cosine cao).

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Áp dụng công thức: num_chunks = ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.11)
> Đáp án: 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên vì bước nhảy (stride) giữa các chunk bị rút ngắn lại. Việc tăng overlap giúp đảm bảo các thông tin quan trọng nằm ở ranh giới giữa hai chunk không bị mất ngữ cảnh (context).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Y khoa & Di truyền học (Genomics, Cancer Screening, Thalassemia)

**Tại sao nhóm chọn domain này?**
> Đây là lĩnh vực có lượng kiến thức chuyên môn đồ sộ, yêu cầu sự chính xác cao và thường xuyên có các tài liệu hướng dẫn (Action Guide) phức tạp. Việc dùng RAG giúp nhân viên y tế và bệnh nhân truy xuất nhanh thông tin chẩn đoán và điều trị.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 01_Prenatal_Genome_White_Paper | Link: https://prenatalgenome.it/pdf/Pre.. | 13,803 | category: NIPT, date: none |
| 2 | 06_Non_Invasive_Prenatal_Testing | Link: https://www.genetics.edu.au/PDF/.. | 4,709 | category: NIPT, date: 2021 |
| 3 | 02_Alpha_Thalassemia_Fact_Sheet | Link: https://static1.squarespace.com/.. | 7,504 | category: alpha thalassamia, date: 2022 |
| 4 | 04_Brain_Tumours_Factsheet | Link: https://www.cclg.org.uk/sites/de.. | 11,238 | category: Brain tumor, date: 2022 |
| 5 | 03_Mendelian_Inheritance_Lecture | Link: https://uomus.edu.iq/img/lectur.. | 5,718 | category: Medelian inheritance, date: 2023 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | data/filename.md | Truy xuất lại URL gốc từ các tổ chức y tế. |
| category | string | NIPT, Brain tumor | Phân loại bệnh lý để lọc kết quả chính xác. |
| date | string | 2022, 2023 | Đảm bảo thông tin y khoa mới nhất. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Alpha-Thalassemia Fact Sheet | FixedSizeChunker (`fixed_size`) | 17 | 488.47 | Kém (thường xuyên bị cắt ngang câu/đầu mục) |
| Alpha-Thalassemia Fact Sheet | SentenceChunker (`by_sentences`) | 18 | 414.44 | Trung bình (câu nguyên vẹn nhưng dễ mất ngữ cảnh đoạn) |
| Alpha-Thalassemia Fact Sheet | RecursiveChunker (`recursive`) | 69 | 106.77 | Tốt (tôn trọng các Heading và cấu trúc danh sách) |

### Strategy Của Tôi

**Loại:** RecursiveChunker (Chiến lược đệ quy)

**Mô tả cách hoạt động:**
> RecursiveChunker hoạt động bằng cách cố gắng chia nhỏ văn bản dựa trên một danh sách các dấu phân cách có thứ tự ưu tiên giảm dần (ví dụ: chia theo đoạn `\n\n`, rồi đến dòng `\n`, rồi đến câu `. `, và cuối cùng là từ ` `). Nếu một đoạn văn bản sau khi chia vẫn vượt quá kích thước `chunk_size` tối đa (ví dụ 300 ký tự), nó sẽ tiếp tục dùng dấu phân cách ưu tiên tiếp theo để chia nhỏ thêm bằng kỹ thuật đệ quy.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Các tài liệu Y khoa (như Fact Sheet hay White Paper) thường có cấu trúc phân cấp rất rõ ràng với các tiêu đề (Heading), danh sách (Bullet points) và đoạn văn. Việc dùng RecursiveChunker giúp tôn trọng các cấu trúc tự nhiên này, giữ trọn vẹn ngữ cảnh của từng đoạn kiến thức thay vì cắt ngang tùy tiện như FixedSizeChunker.

**Code snippet (nếu custom):**
> (Ghi chú: Tôi sử dụng class `RecursiveChunker` đã implement trong file `src/chunking.py`, không phải là custom external script).

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Alpha-Thalassemia Fact Sheet | SentenceChunker (Best Baseline) | 18 | 414.44 | Khá (đôi khi chunk quá dài làm loãng context) |
| Alpha-Thalassemia Fact Sheet | **RecursiveChunker (Của tôi)** | 69 | 106.77 | Tuyệt vời (truy xuất cực nhanh và chính xác) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Quách Ngọc Quang (Tôi) | RecursiveChunker (cs=300) | 9 | Tôn trọng cấu trúc phân cấp, giữ ngữ cảnh y khoa tốt. | Số lượng chunk lớn (69), tốn tài nguyên store. |
| Khổng Mạnh Tuấn | fixed_size | 8 | Ổn định, dễ kiểm soát chunk. | Query 4/5 vẫn lệch tài liệu kỳ vọng, chỉ trả lời được câu hỏi dễ, các câu hỏi phức tạp chọn đúng tài liệu và chỉ số liên quan cao nhưng câu trả lời không chính xác (model local). |
| Lâm Hoàng Hải | sentence | 6 | Dễ cài đặt, không bị cắt giữa đoạn. | Đúng 3/5 query, miss ý của những câu hỏi phức tạp, cần suy luận từ tương đồng, ngữ cảnh. |
| Nguyễn Hoàng Long | RecursiveChunker (chunk_size=500) | 8 | Giữ ngữ cảnh theo paragraph, phù hợp tài liệu y khoa có cấu trúc. | Chunks có thể quá lớn cho factsheet câu ngắn. |
| Thuận | RecursiveChunker | 8 | Fallback thông minh, tôn trọng cấu trúc tự nhiên của tài liệu và rất linh hoạt. | Không thể xử lí bảng biểu, không có overlap. |
| Trần Thái Huy | SentenceChunker | 7 | Dễ cài đặt, chunk theo câu dễ đọc. | Tạo nhiều chunk hơn, dễ miss ý ở câu hỏi cần ngữ cảnh dài. |
| Nguyễn Mạnh Dũng | RecursiveChunker (chunk_size=600) | 8 | Data của nhóm có cấu trúc thứ bậc nên ngữ cảnh được giữ lại tốt. | Chunk lớn đôi khi làm loãng ngữ cảnh cho fact ngắn. |

**Chiến lược nào nhóm bạn chọn làm "mặc định"? Tại sao?**
> Nhóm chọn **Recursive Chunker** vì nó mang lại sự linh hoạt cao nhất, giúp giữ cho các chunk có độ dài đồng đều nhưng vẫn tôn trọng ranh giới tự nhiên của văn bản (đoạn, câu, từ).

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng Regex `(?<=[.!?])\s+|(?<=\.)\n` để xác định ranh giới câu một cách chính xác. Sau đó, gom các câu thành từng khối dựa trên `max_sentences_per_chunk` và làm sạch khoảng trắng bằng `strip()`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng thuật toán đệ quy để ưu tiên tách văn bản từ các dấu phân cách thô (như đoạn văn) đến tinh (như từ). Nếu một mẩu văn bản sau khi tách vẫn lớn hơn `chunk_size`, hàm sẽ tự gọi lại chính nó với các dấu phân tách tiếp theo trong danh sách ưu tiên.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Sử dụng cơ chế hybrid hỗ trợ cả ChromaDB và In-memory. Với mỗi Document, hệ thống băm nhỏ thành chunk, chuyển thành vector qua `embedding_fn` và lưu trữ dưới dạng Dictionary chứa đầy đủ Content, Vector và Metadata.

**`search_with_filter` + `delete_document`** — approach:
> Thực hiện lọc metadata trước khi tính toán độ tương đồng (pre-filtering) để tối ưu hiệu năng. Hàm xóa được thiết kế để tìm kiếm chính xác `doc_id` bên trong metadata của từng chunk để loại bỏ triệt để dữ liệu liên quan.

### KnowledgeBaseAgent

**`answer`** — approach:
> Áp dụng mô hình RAG: Truy xuất Top-K mẩu tin liên quan, gộp chúng vào Prompt làm ngữ cảnh (Context) trước khi gửi yêu cầu cho LLM, giúp AI trả lời chính xác và tránh ảo giác.

### Test Results

```
============================= 42 passed in 0.27s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Học máy rất thú vị" | "Machine learning is fun" | High | 0.0534 | Một phần |
| 2 | "Tôi yêu lập trình" | "Tôi thích viết mã" | High | 0.5672 | Đúng |
| 3 | "Mặt trời mọc ở đằng đông" | "Tôi đang ăn phở" | Low | 0.6429 | Bất ngờ |
| 4 | "Ngày mai trời mưa" | "Ngày mai có mưa"| High | 0.8694 | Đúng |
| 5 | "Chào bạn" | "Chào bạn" | High | 1.0000 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là các câu có ý nghĩa giống hệt nhau (Cặp 1, 2) lại có điểm tương đồng cực thấp. Điều này xảy ra vì tôi đang sử dụng Mock Embedder (dựa trên thuật toán băm Hash MD5). Nó cho thấy rằng các vector ngẫu nhiên hoặc dựa trên mã băm không thể biểu diễn được "ngữ nghĩa" của ngôn ngữ; chúng chỉ nhận diện được sự trùng khớp chính xác tuyệt đối của chuỗi ký tự.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | In NIPT, what is the role of paternal DNA information? | Analysed to estimate fetal fraction and confirm fetal DNA. |
| 2 | What genetic factor determines the subtype of alpha-thalassemia? | The number of damaged or missing alpha-globin genes. |
| 3 | What is the basic human chromosome makeup? | 46 chromosomes (22 pairs autosomes + 2 sex chromosomes). |
| 4 | What is the most common malignant brain tumour in children? | Medulloblastoma. |
| 5 | Why can brain tumours cause headaches and seizures? | Raised pressure inside the head or blocking fluid flow. |

### Kết Quả Của Tôi (Dùng all-MiniLM-L6-v2 - FINAL)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Paternal DNA in NIPT | NIPT Fact Sheet | 0.5207 | Yes | [Final Model] DNA từ cha giúp xác định... |
| 2 | Alpha-thalassemia factor | Alpha-thala Fact Sheet | 0.6747 | Yes | [Final Model] Dựa trên số lượng gen hỏng... |
| 3 | Human chromosome makeup | PrenatalGenome WhitePaper | 0.3586 | Yes | [Final Model] Bao gồm 46 nhiễm sắc thể... |
| 4 | Child brain tumour | Brain Tumours Factsheet | 0.6075 | Yes | [Final Model] Medulloblastoma rất phổ biến... |
| 5 | Headache/Seizure cause | Brain Tumours Factsheet | 0.5726 | Yes | [Final Model] Do áp lực nội sọ tăng cao... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

### Hiệu Quả Của Metadata Filtering (Part 3 Highlight)
Để đáp ứng yêu cầu của Part 3, tôi đã thực hiện một truy vấn có sử dụng bộ lọc Metadata:
- **Query:** "symptoms and headaches"
- **Filter:** `category='Brain tumor'`
- **Kết quả:** Hệ thống đã bỏ qua toàn bộ 13 tài liệu khác và tìm chính xác đoạn văn trong file kiến thức về Khối u não (`cclg-brain-tumours-factsheet`). Điều này chứng minh rằng việc gán Metadata từ file CSV giúp hệ thống RAG thu hẹp phạm vi tìm kiếm và loại bỏ nhiễu cực kỳ hiệu quả, ngay cả khi dùng Mock Embedder.

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được từ Long về việc tối ưu `chunk_size`. Trong khi tôi dùng size 300 để tập trung vào các câu trả lời ngắn, Long dùng size 500 để giữ được ngữ cảnh trọn vẹn của các đoạn văn (paragraph). Điều này gợi ý rằng chúng ta có thể kết hợp nhiều kích thước chunk khác nhau cho cùng một tài liệu để tăng hiệu quả truy xuất.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tập trung hơn vào việc xử lý tiền dữ liệu (cleaning) để loại bỏ các ký tự đặc biệt trước khi đưa vào chunking, điều này sẽ giúp các vector embedding phản ánh ý nghĩa ngữ nghĩa thuần khiết hơn. Đồng thời, tôi sẽ mở rộng schema metadata để bao gồm cả `author` hoặc `department` nhằm tăng độ chính xác của bộ lọc.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **88 / 100** |
