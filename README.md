# 🤖 RAG Question-Answering Chatbot with Pinecone & Google Gemini

Hệ thống Chatbot hỏi - đáp thông minh áp dụng kiến trúc **RAG (Retrieval-Augmented Generation)**, kết hợp giữa cơ sở dữ liệu vector **Pinecone Serverless** và mô hình AI thế hệ mới **Google Gemini**. Hệ thống cho phép tìm kiếm ngữ cảnh chính xác từ kho tài liệu video YouTube và sinh câu trả lời kèm trích dẫn nguồn uy tín.

---

## 📑 Mục lục
- [Kiến trúc hoạt động (RAG Architecture)](#-kiến-trúc-hoạt-động-rag-architecture)
- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống & Thư viện](#-yêu-cầu-hệ-thống--thư-viện)
- [Hướng dẫn cài đặt từng bước](#-hướng-dẫn-cài-đặt-từng-bước)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Minh họa kết quả](#-minh-họa-kết-quả)

---

## 🏗 Kiến trúc hoạt động (RAG Architecture)

Quy trình xử lý của hệ thống tuân theo mô hình RAG tiêu chuẩn:

```
[Dữ liệu Video YouTube (CSV)] 
       │
       ▼
 [Google Gemini Embedding] (gemini-embedding-001, 1536 dims)
       │
       ▼
 [Pinecone Vector Database] (Index: pinecone-datacamp, Serverless)
       ▲
       │  (Similarity Search: Cosine Top-k)
       │
 [Câu hỏi người dùng] ──► [Embed Query]
                               │
                               ▼
                    [Top Chunks + Context]
                               │
                               ▼
                      [Google Gemini 3.5]
                               │
                               ▼
             [Câu trả lời kèm Trích dẫn nguồn (Sources)]
```

1. **Ingestion (Nạp dữ liệu):**
   - Đọc dữ liệu từ file `youtube_rag_data.csv` (gồm ID, text, video title, url, timestamp).
   - Tạo vector nhúng 1536 chiều bằng `gemini-embedding-001`.
   - Lưu trữ các vector kèm metadata vào Pinecone Serverless Index.

2. **Retrieval (Truy xuất):**
   - Chuyển câu hỏi của người dùng thành vector tương ứng.
   - Truy vấn Pinecone để tìm ra $k$ đoạn văn bản có độ tương đồng Cosine cao nhất.

3. **Generation (Sinh câu trả lời):**
   - Đóng gói câu hỏi và ngữ cảnh trích xuất vào Prompt có kiểm soát.
   - Gửi sang mô hình `gemini-3.5-flash-lite` để tổng hợp câu trả lời khách quan, kèm đường dẫn video nguồn.

---

## ✨ Tính năng nổi bật

- ⚡ **Hoàn toàn miễn phí:** Sử dụng **Google Gemini API** (Free Tier) cho cả embedding và sinh văn bản, không tốn phí nạp thẻ.
- 🚀 **Tìm kiếm vector tốc độ cao:** Sử dụng **Pinecone Serverless** tối ưu hiệu năng tìm kiếm tương đồng trên đám mây (AWS us-east-1).
- 💬 **Giao diện dòng lệnh tương tác (Interactive CLI):** Chat trực tiếp trên terminal với cơ chế lặp liên tục, gõ `exit` để thoát.
- 🔗 **Trích dẫn nguồn minh bạch:** Mỗi câu trả lời đều đi kèm tiêu đề video và liên kết YouTube trực tiếp đến đúng nội dung.
- 🛡 **Cơ chế chống Rate Limit thông minh:** Tự động chia nhỏ batch khi nạp dữ liệu và có cơ chế auto-retry khi chạm ngưỡng Free Tier.
- ⚡ **Lazy Loading Connection:** Tối ưu thời gian khởi động app tức thì, chỉ kết nối mạng đến Pinecone khi thực hiện truy vấn.

---

## 📂 Cấu trúc dự án

```text
Pinecone/
├── Slide/                      # Slide bài giảng và tài liệu học tập
│   ├── chapter1 (2).pdf
│   ├── chapter2 (2).pdf
│   └── chapter3 (3).pdf
├── .env.example                # Mẫu cấu hình biến môi trường
├── .gitignore                  # Bỏ qua file nhạy cảm (.env, .venv, cache)
├── rag_chatbot.py              # Mã nguồn chính của RAG Chatbot
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── youtube_rag_data.csv        # Dữ liệu mẫu bài giảng YouTube
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## 🛠 Yêu cầu hệ thống & Thư viện

- **Python:** 3.10 trở lên
- **Thư viện chính:**
  - `pinecone`: SDK chính thức làm việc với Pinecone Vector Database.
  - `google-genai`: SDK mới nhất từ Google làm việc với Gemini API.
  - `pandas`, `numpy`: Xử lý và tính toán ma trận, bảng dữ liệu.
  - `python-dotenv`: Quản lý biến môi trường an toàn từ file `.env`.

---

## 🚀 Hướng dẫn cài đặt từng bước

### Bước 1: Clone kho mã nguồn về máy
```bash
git clone https://github.com/Hoanglovecode/--RAG--Chatbot--With--Pinecone--.git
cd --RAG--Chatbot--With--Pinecone--
```

### Bước 2: Tạo và kích hoạt môi trường ảo
- **Trên Windows:**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### Bước 4: Thiết lập biến môi trường API Key
1. Tạo một file `.env` ở thư mục gốc (dựa trên mẫu `.env.example`):
   ```bash
   cp .env.example .env
   ```
2. Điền các khóa API của bạn vào file `.env`:
   ```env
   PINECONE_API_KEY=pcsk_your_pinecone_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   > 🔑 **Lấy API Key ở đâu?**
   > - **Pinecone:** Đăng ký tài khoản miễn phí tại [Pinecone Console](https://app.pinecone.io/) và copy API Key.
   > - **Gemini:** Nhận API Key miễn phí tức thì tại [Google AI Studio](https://aistudio.google.com/).

---

## 📖 Hướng dẫn sử dụng

### 1. Khởi động Chatbot (Chế độ tương tác mặc định)
Chỉ cần chạy lệnh sau, chatbot sẽ mở giao diện đối thoại trực tiếp trên terminal:
```bash
python rag_chatbot.py
```
*Nhập câu hỏi bất kỳ, chatbot sẽ truy vấn ngữ cảnh và trả lời. Gõ `exit` hoặc `quit` khi muốn thoát.*

### 2. Đặt câu hỏi nhanh từ Terminal
Bạn có thể hỏi ngay mà không cần vào chế độ tương tác:
```bash
python rag_chatbot.py "How to choose indexes for similarity search in Faiss?"
```

### 3. Nạp dữ liệu vào Pinecone (Ingestion)
Nếu bạn cập nhật file `youtube_rag_data.csv` mới hoặc muốn nạp lại vector vào Pinecone:
```bash
python rag_chatbot.py ingest
```

---

## 💡 Minh họa kết quả

```text
=======================================================
🤖 CHÀO MỪNG ĐẾN VỚI RAG CHATBOT (Pinecone + Google Gemini)
💡 Nhập câu hỏi của bạn bên dưới (hoặc gõ 'exit' để thoát)
=======================================================

Bạn: How to choose indexes for similarity search in Faiss?

🔍 Đang tìm kiếm tài liệu liên quan từ Pinecone...
🤖 Gemini đang xử lý câu trả lời...

==================================================
Based on the provided context, you can decide which index to use based on your data and your specific use case (for example, choosing between higher speed or higher quality). The context notes that the indexes range from flat indexes (which provide plain and simple, 100% search quality) to others that offer a 50-50 trade-off, depending on whether you need more speed or higher quality.

Sources:
- Choosing Indexes for Similarity Search (Faiss in Python): https://youtu.be/B7wmo_NImgM
- Choosing Indexes for Similarity Search (Faiss in Python): https://youtu.be/B7wmo_NImgM
- Choosing Indexes for Similarity Search (Faiss in Python): https://youtu.be/B7wmo_NImgM
==================================================
```

---

## 📜 Giấy phép
Dự án được phân phối dưới giấy phép **MIT License**. Bạn được tự do học tập, sử dụng và phát triển mở rộng.
