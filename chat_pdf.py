import sys
import os
import fitz  # PyMuPDF
import psycopg2
import requests
from rank_bm25 import BM25Okapi

# Helper function to chunk text
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    res = requests.post(
        "http://localhost:11434/api/embeddings", 
        json={"model": "nomic-embed-text", "prompt": text}
    )
    if res.status_code != 200:
        raise Exception(f"Ollama embedding error: {res.text}")
    return res.json()["embedding"]

# Simple tokenization for BM25 (lowercase and split by space)
def tokenize(text):
    return text.lower().split()

# Connect to Database
try:
    conn = psycopg2.connect("dbname=pdf_rag user=ubuntu port=5433")
    cur = conn.cursor()
except Exception as e:
    print(f"Database connection error: {e}")
    sys.exit(1)

print("=" * 70)
print("📚 INTERACTIVE HYBRID RAG (PYTHON BM25 + PGVECTOR)")
print("=" * 70)

# 1. Ask if the user wants to ingest a new PDF
pdf_file = input("Enter the name of a new PDF to ingest (or press Enter to skip): ").strip()

if pdf_file:
    if not os.path.exists(pdf_file):
        print(f"Error: File '{pdf_file}' not found. Make sure you used scp to upload it!")
        sys.exit(1)
        
    print(f"\n[1/3] Clearing old data from the database...")
    cur.execute("TRUNCATE TABLE documents;")
    conn.commit()

    print(f"[2/3] Processing '{pdf_file}' (Extracting, OCR, and Chunking)...")
    doc = fitz.open(pdf_file)
    inserted_count = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text().strip()
        
        # OCR Fallback
        if not page_text:
            try:
                tp = page.get_textpage_ocr(language="eng")
                page_text = page.get_text(textpage=tp).strip()
            except:
                pass

        if not page_text:
            continue

        text_chunks = chunk_text(page_text, chunk_size=500, overlap=100)
        
        for chunk in text_chunks:
            if not chunk.strip():
                continue
            embedding = get_embedding(chunk)
            cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (chunk, embedding))
            inserted_count += 1
            
    conn.commit()
    print(f"[3/3] Ingestion complete! Successfully saved {inserted_count} chunks.")
    print("=" * 70)

# 2. Build BM25 Index in Memory
print("\nLoading chunks into Python for BM25 indexing...")
cur.execute("SELECT id, content FROM documents;")
all_docs = cur.fetchall()

doc_ids = [row[0] for row in all_docs]
doc_contents = [row[1] for row in all_docs]
tokenized_corpus = [tokenize(doc) for doc in doc_contents]

# Initialize BM25 algorithm
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 Index successfully built with {len(doc_contents)} chunks!")

# 3. Interactive Chat Loop
print("\nReady for questions! (Type 'exit' to quit)")
while True:
    try:
        query = input("\nAsk a question: ").strip()
        if not query: continue
        if query.lower() in ["exit", "quit"]: break

        # --- A. DENSE VECTOR SEARCH (PostgreSQL) ---
        query_emb = get_embedding(query)
        cur.execute("""
            SELECT id, RANK() OVER (ORDER BY embedding <=> %s::vector) AS v_rank
            FROM documents 
            ORDER BY embedding <=> %s::vector LIMIT 10;
        """, (query_emb, query_emb))
        vector_results = cur.fetchall()
        vector_ranks = {row[0]: row[1] for row in vector_results}

        # --- B. SPARSE KEYWORD SEARCH (Python BM25) ---
        tokenized_query = tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Pair IDs with scores, filter out zero-matches, and sort descending
        id_score_pairs = [(doc_ids[i], bm25_scores[i]) for i in range(len(doc_ids))]
        matched_pairs = [pair for pair in id_score_pairs if pair[1] > 0]
        matched_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Assign rank positions (1 to 10)
        keyword_ranks = {}
        for rank, (doc_id, score) in enumerate(matched_pairs[:10], start=1):
            keyword_ranks[doc_id] = rank

        # --- C. RECIPROCAL RANK FUSION (Python Math) ---
        all_retrieved_ids = set(vector_ranks.keys()).union(set(keyword_ranks.keys()))
        
        rrf_scores = []
        for doc_id in all_retrieved_ids:
            v_rank = vector_ranks.get(doc_id, None)
            k_rank = keyword_ranks.get(doc_id, None)
            
            # Standard RRF formula: 1 / (60 + Rank)
            v_score = 1.0 / (60 + v_rank) if v_rank else 0.0
            k_score = 1.0 / (60 + k_rank) if k_rank else 0.0
            total_rrf = v_score + k_score
            
            rrf_scores.append((doc_id, total_rrf, v_rank, k_rank))
            
        # Sort by final RRF score and grab top 3
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = rrf_scores[:3]

        # --- DISPLAY RESULTS ---
        print("\n" + "-" * 60)
        print("🔍 RECIPROCAL RANK FUSION (BM25 + PGVECTOR)")
        print("-" * 60)
        
        context_chunks = []
        for i, (doc_id, rrf_score, v_rank, k_rank) in enumerate(top_3):
            # Fetch content directly from our Python list instead of querying DB again
            idx = doc_ids.index(doc_id)
            chunk_text = doc_contents[idx]
            context_chunks.append(chunk_text)
            
            v_disp = v_rank if v_rank is not None else "N/A"
            k_disp = k_rank if k_rank is not None else "N/A"
            v_math = f"(1/{60 + v_rank})" if v_rank is not None else "0.0"
            k_math = f"(1/{60 + k_rank})" if k_rank is not None else "0.0"
            
            print(f"🏆 Final Rank #{i+1} | RRF Score: {rrf_score:.5f}")
            print(f"   ▶ Vector Match Rank: {v_disp}  -> {v_math}")
            print(f"   ▶ BM25 Keyword Rank: {k_disp}  -> {k_math}")
            print(f"   [ {len(chunk_text)} chars ] : {chunk_text[:100]}...\n")
            
        if not context_chunks:
            print("No relevant context found. Try a different query.")
            continue
            
        print("-" * 60)
        print("Generating answer with Llama 3.2 based on these top chunks...")
        
        # --- D. LLM GENERATION ---
        context = "\n\n".join(context_chunks)
        prompt = f"Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information, answer the query: {query}"

        gen_res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False})
        answer = gen_res.json()["response"]
        
        print("\n" + "=" * 70)
        print("FINAL ANSWER:")
        print("=" * 70)
        print(answer)
        print("=" * 70)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")

cur.close()
conn.close()
