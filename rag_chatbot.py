"""
RAG chatbot voi Pinecone + OpenAI (theo DataCamp - Building AI Applications with Pinecone, chuong 3)

Cach chay:
    python rag_chatbot.py ingest                     # nap du lieu vao Pinecone (chay 1 lan)
    python rag_chatbot.py ask "How to build next-level Q&A with OpenAI"
    python rag_chatbot.py                            # chay cau hoi mac dinh
"""
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------- Cau hinh ----------
INDEX_NAME = "pinecone-datacamp"
NAMESPACE = "youtube_rag_dataset"
EMB_MODEL = "text-embedding-3-small"   # 1536 chieu
CHAT_MODEL = "gpt-4o-mini"
CSV_PATH = "youtube_rag_data.csv"
BATCH_LIMIT = 100

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# ---------- 1. Tao index (neu chua co) va ket noi ----------
def get_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(INDEX_NAME)


index = get_index()


# ---------- 2. Nap tai lieu (ingest) ----------
def ingest():
    df = pd.read_csv(CSV_PATH)
    n_batches = max(1, len(df) // BATCH_LIMIT)

    for batch in np.array_split(df, n_batches):
        metadatas = [
            {
                "text_id": int(row["id"]),
                "text": row["text"],
                "title": row["title"],
                "url": row["url"],
                "published": str(row["published"]),  # metadata khong nhan datetime
            }
            for _, row in batch.iterrows()
        ]
        texts = batch["text"].tolist()
        # ID co dinh theo id dong -> chay lai khong bi trung lap
        ids = [f"yt-{int(i)}" for i in batch["id"]]

        response = client.embeddings.create(input=texts, model=EMB_MODEL)
        embeds = [d.embedding for d in response.data]

        index.upsert(vectors=list(zip(ids, embeds, metadatas)), namespace=NAMESPACE)

    print(index.describe_index_stats())


# ---------- 3. Ham truy xuat ----------
def retrieve(query, top_k, namespace, emb_model):
    query_response = client.embeddings.create(input=query, model=emb_model)
    query_emb = query_response.data[0].embedding

    retrieved_docs = []
    sources = []
    docs = index.query(
        vector=query_emb, top_k=top_k, namespace=namespace, include_metadata=True
    )
    for doc in docs["matches"]:
        retrieved_docs.append(doc["metadata"]["text"])
        sources.append((doc["metadata"]["title"], doc["metadata"]["url"]))
    return retrieved_docs, sources


# ---------- 4. Tao prompt co ngu canh ----------
def prompt_with_context_builder(query, docs):
    delim = "\n\n---\n\n"
    prompt_start = "Answer the question based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    return prompt_start + delim.join(docs) + prompt_end


# ---------- 5. Hoi - dap ----------
def question_answering(prompt, sources, chat_model):
    sys_prompt = "You are a helpful assistant that always answers questions."

    res = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    answer = res.choices[0].message.content.strip()
    answer += "\n\nSources:"
    for source in sources:
        answer += "\n" + source[0] + ": " + source[1]
    return answer


# ---------- 6. Ghep tat ca ----------
def ask(query):
    documents, sources = retrieve(
        query, top_k=3, namespace=NAMESPACE, emb_model=EMB_MODEL
    )
    prompt_with_context = prompt_with_context_builder(query, documents)
    print(prompt_with_context)
    print("\n" + "=" * 60 + "\n")

    answer = question_answering(
        prompt=prompt_with_context, sources=sources, chat_model=CHAT_MODEL
    )
    print(answer)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) > 2 and sys.argv[1] == "ask":
        ask(sys.argv[2])
    else:
        ask("How to build next-level Q&A with OpenAI")
