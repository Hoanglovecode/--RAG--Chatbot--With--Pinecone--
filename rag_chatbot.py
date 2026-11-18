"""
RAG Chatbot kết hợp Pinecone + Google Gemini.

Cách chạy:
    python rag_chatbot.py                            # khởi động chatbot hỏi - đáp tương tác trực tiếp
    python rag_chatbot.py "câu hỏi của bạn"          # hỏi nhanh 1 câu từ terminal
    python rag_chatbot.py ingest                     # nạp dữ liệu từ youtube_rag_data.csv vào Pinecone (chạy 1 lần)
"""
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------- Kiểm tra API Key ----------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not PINECONE_API_KEY:
    print("[LỖI] Thiếu PINECONE_API_KEY trong file .env! Vui lòng cấu hình trước khi chạy.")
    sys.exit(1)

if not GEMINI_API_KEY:
    print("[LỖI] Thiếu GEMINI_API_KEY trong file .env! Vui lòng cấu hình trước khi chạy.")
    sys.exit(1)

# ---------- Cấu hình ----------
INDEX_NAME = "pinecone-datacamp"
NAMESPACE = "gemini_youtube_rag"       # Namespace riêng cho vector tạo bởi Gemini
EMB_MODEL = "gemini-embedding-001"     # Mô hình embedding của Google
EMB_DIM = 1536                         # Đồng bộ 1536 chiều với index Pinecone
CHAT_MODEL = "gemini-3.5-flash-lite"   # Mô hình LLM trả lời nhanh và miễn phí
CSV_PATH = "youtube_rag_data.csv"
BATCH_LIMIT = 25                       # Batch kích thước 25 để tránh vượt Rate Limit Free Tier

# Khởi tạo clients
client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# ---------- 1. Tạo index (nếu chưa có) và kết nối ----------
_index = None

def get_index():
    global _index
    if _index is None:
        existing_indexes = pc.list_indexes().names()
        if INDEX_NAME not in existing_indexes:
            print(f"Index '{INDEX_NAME}' chưa tồn tại, đang tạo mới...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMB_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index(INDEX_NAME)
    return _index


# ---------- 2. Hàm tạo Embeddings từ Gemini ----------
def get_batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Tạo vector embeddings cho danh sách văn bản qua Gemini có tự động retry khi chạm giới hạn."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = client.models.embed_content(
                model=EMB_MODEL,
                contents=texts,
                config={"output_dimensionality": EMB_DIM},
            )
            return [e.values for e in res.embeddings]
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = 12
                print(f"  [Rate limit Free Tier] Đang tạm nghỉ {wait_time}s rồi tiếp tục...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Vượt quá số lần thử lại tạo embedding.")


def get_query_embedding(query: str) -> list[float]:
    """Tạo vector embedding cho một câu truy vấn."""
    res = client.models.embed_content(
        model=EMB_MODEL,
        contents=query,
        config={"output_dimensionality": EMB_DIM},
    )
    return res.embeddings[0].values


# ---------- 3. Nạp tài liệu (ingest) ----------
def ingest(max_rows: int = 200):
    """
    Đọc dữ liệu từ CSV và đưa vào Pinecone.
    Mặc định nạp 200 dòng mẫu đầu tiên (đủ cho demo và tránh chạm rate limit).
    """
    if not os.path.exists(CSV_PATH):
        print(f"[LỖI] Không tìm thấy file dữ liệu: {CSV_PATH}")
        return

    print(f"Đang đọc dữ liệu từ '{CSV_PATH}'...")
    df = pd.read_csv(CSV_PATH)
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        print(f"Sẽ nạp {len(df)} dòng dữ liệu mẫu vào namespace '{NAMESPACE}'...")

    total_rows = len(df)
    index = get_index()
    for start_idx in range(0, total_rows, BATCH_LIMIT):
        batch = df.iloc[start_idx : start_idx + BATCH_LIMIT]
        metadatas = [
            {
                "text_id": str(row["id"]),
                "text": str(row["text"]),
                "title": str(row["title"]),
                "url": str(row["url"]),
                "published": str(row["published"]),
            }
            for _, row in batch.iterrows()
        ]
        texts = [str(t) for t in batch["text"].tolist()]
        ids = [f"yt-{str(row_id)}" for row_id in batch["id"]]

        # Gọi Gemini tạo embedding
        embeds = get_batch_embeddings(texts)

        # Upsert vào Pinecone
        index.upsert(vectors=list(zip(ids, embeds, metadatas)), namespace=NAMESPACE)
        print(f"Đã tải lên: {min(start_idx + BATCH_LIMIT, total_rows)}/{total_rows} bản ghi...")
        time.sleep(1)  # Giãn cách nhẹ để giữ an toàn quota Free Tier

    print("\n--- Thống kê Index sau khi nạp ---")
    print(index.describe_index_stats())
    print("\n[THÀNH CÔNG] Dữ liệu đã sẵn sàng! Bây giờ bạn có thể đặt câu hỏi.")


# ---------- 4. Hàm truy xuất (Retrieve) ----------
def retrieve(query: str, top_k: int = 3):
    query_emb = get_query_embedding(query)
    index = get_index()

    retrieved_docs = []
    sources = []
    docs = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
    )

    for match in docs.get("matches", []):
        meta = match.get("metadata", {})
        if "text" in meta:
            retrieved_docs.append(meta["text"])
        title = meta.get("title", "Unknown")
        url = meta.get("url", "")
        sources.append((title, url))

    return retrieved_docs, sources


# ---------- 5. Tạo prompt có ngữ cảnh ----------
def prompt_with_context_builder(query: str, docs: list[str]) -> str:
    delim = "\n\n---\n\n"
    prompt_start = (
        "Answer the question based on the context below. "
        "If you do not know the answer from the context, state that clearly.\n\n"
        "Context:\n"
    )
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    return prompt_start + delim.join(docs) + prompt_end


# ---------- 6. Hỏi - đáp (LLM Generation) ----------
def question_answering(prompt: str, sources: list[tuple[str, str]]) -> str:
    interaction = client.interactions.create(
        model=CHAT_MODEL,
        input=prompt,
    )
    answer = interaction.output_text or ""

    if sources:
        answer += "\n\nSources:"
        for title, url in sources:
            answer += f"\n- {title}: {url}"
    return answer


# ---------- 7. Ghép toàn bộ quy trình RAG ----------
def ask(query: str):
    print("\n🔍 Đang tìm kiếm tài liệu liên quan từ Pinecone...")
    documents, sources = retrieve(query, top_k=3)

    if not documents:
        print(f"\n[CHÚ Ý] Chưa tìm thấy dữ liệu trong namespace '{NAMESPACE}'.")
        print("Vì bạn vừa chuyển sang mô hình Gemini, bạn cần nạp dữ liệu một lần bằng lệnh:")
        print("    python rag_chatbot.py ingest\n")
        return

    print("🤖 Gemini đang xử lý câu trả lời...")
    prompt_with_context = prompt_with_context_builder(query, documents)
    answer = question_answering(prompt_with_context, sources)
    print("\n" + "=" * 50)
    print(answer)
    print("=" * 50)


if __name__ == "__main__":
    # 1. Nếu có tham số 'ingest': nạp dữ liệu vào Pinecone
    if len(sys.argv) > 1 and sys.argv[1].lower() == "ingest":
        ingest()
    # 2. Nếu người dùng truyền sẵn câu hỏi qua dòng lệnh (ví dụ: python rag_chatbot.py "câu hỏi")
    elif len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        ask(user_query)
    # 3. Mặc định: Người dùng nhập câu hỏi trực tiếp trên terminal
    else:
        print("=" * 55)
        print("🤖 CHÀO MỪNG ĐẾN VỚI RAG CHATBOT (Pinecone + Google Gemini)")
        print("💡 Nhập câu hỏi của bạn bên dưới (hoặc gõ 'exit' để thoát)")
        print("=" * 55)

        while True:
            try:
                question = input("\nBạn: ").strip()
                if not question:
                    continue
                if question.lower() in ["exit", "quit", "q"]:
                    print("Tạm biệt bạn! 👋")
                    break
                ask(question)
            except (KeyboardInterrupt, EOFError):
                print("\nĐã thoát chương trình. Tạm biệt! 👋")
                break

