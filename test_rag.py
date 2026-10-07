import sys
import psycopg2
import requests

# 1. Connect to ShaktiDB
try:
    conn = psycopg2.connect("dbname=pdf_rag user=ubuntu port=5433")
    cur = conn.cursor()
except Exception as e:
    print(f"Database connection error: {e}")
    sys.exit(1)

print("=" * 60)
print("Interactive Hybrid RAG Terminal Initialized!")
print("Type your questions below. Type 'exit' or 'quit' to stop.")
print("=" * 60)

while True:
    try:
        query = input("\nAsk a question: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit"]:
            print("Exiting RAG session. Goodbye!")
            break

        # 2. Get embedding for the query from Ollama
        res = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": query}
        )
        if res.status_code != 200:
            print(f"Ollama embedding error: {res.text}")
            continue
        query_emb = res.json()["embedding"]

        # 3. Run Hybrid Search (Vector + Keyword via RRF)
        rrf_sql = """
        WITH vector_search AS (
            SELECT id, content, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
            FROM documents ORDER BY embedding <=> %s::vector LIMIT 10
        ),
        keyword_search AS (
            SELECT id, content, RANK() OVER (ORDER BY ts_rank(fts_tokens, plainto_tsquery('english', %s)) DESC) AS rank
            FROM documents WHERE fts_tokens @@ plainto_tsquery('english', %s) LIMIT 10
        )
        SELECT COALESCE(v.content, k.content) AS content
        FROM vector_search v FULL OUTER JOIN keyword_search k ON v.id = k.id
        ORDER BY (COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) DESC
        LIMIT 3;
        """

        cur.execute(rrf_sql, (query_emb, query_emb, query, query))
        results = cur.fetchall()
        
        # --- NEW SECTION: DISPLAY CHUNKS & VECTORING PROOF ---
        print("\n" + "=" * 50)
        print(f"DATABASE RETRIEVAL: Found {len(results)} relevant chunks")
        print("=" * 50)
        
        context_chunks = []
        for i, row in enumerate(results):
            chunk_text = row[0]
            context_chunks.append(chunk_text)
            print(f"\n[ Chunk {i+1} | Length: {len(chunk_text)} chars ]")
            print(chunk_text)
            
        print("\n" + "=" * 50)
        print("Generating answer with Llama 3.2 based on the above chunks...")
        print("=" * 50)
        # -----------------------------------------------------

        context = "\n\n".join(context_chunks)

        # 4. Generate Answer using Llama 3.2
        prompt = f"Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information, answer the query: {query}"

        gen_res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False}
        )
        
        if gen_res.status_code != 200:
            print(f"Ollama generation error: {gen_res.text}")
            continue

        answer = gen_res.json()["response"]
        
        print("\n" + "-" * 50)
        print("FINAL LLM ANSWER:")
        print("-" * 50)
        print(answer)
        print("-" * 50)

    except KeyboardInterrupt:
        print("\nExiting RAG session. Goodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {e}")

cur.close()
conn.close()
