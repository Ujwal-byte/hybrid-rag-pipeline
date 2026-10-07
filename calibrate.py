import psycopg2
import requests

DB_PARAMS = "dbname=pdf_rag user=ubuntu port=5433"

# 10 test queries: 5 in-document, 5 out-of-document
test_queries = [
    # In-document questions (adjust these to match your PDF)
    ("What is an operating system?", True),
    ("Define process management", True),
    ("What is thread synchronization?", True),
    ("Explain CPU scheduling algorithms", True),
    ("What is a deadlock?", True),
    # Out-of-document questions
    ("How do I bake a chocolate cake?", False),
    ("What is the capital of France?", False),
    ("Explain photosynthesis in plants", False),
    ("Who won the 2022 FIFA World Cup?", False),
    ("How does quantum computing work?", False),
]

def get_embedding(text):
    res = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": text})
    return res.json()["embedding"]

conn = psycopg2.connect(DB_PARAMS)
cur = conn.cursor()

in_doc_scores = []
out_doc_scores = []

print(f"{'Query':<45} | {'Type':<8} | {'Max RRF Score':<12}")
print("-" * 72)

for query, is_in_doc in test_queries:
    query_emb = get_embedding(query)
    rrf_sql = """
    WITH vector_search AS (
        SELECT id, content, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
        FROM documents ORDER BY embedding <=> %s::vector LIMIT 10
    ),
    keyword_search AS (
        SELECT id, content, RANK() OVER (ORDER BY ts_rank(fts_tokens, plainto_tsquery('english', %s)) DESC) AS rank
        FROM documents WHERE fts_tokens @@ plainto_tsquery('english', %s) LIMIT 10
    )
    SELECT (COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) AS rrf_score
    FROM vector_search v FULL OUTER JOIN keyword_search k ON v.id = k.id
    ORDER BY rrf_score DESC LIMIT 1;
    """
    cur.execute(rrf_sql, (query_emb, query_emb, query, query))
    res = cur.fetchone()
    score = res[0] if res else 0.0

    q_type = "In-Doc" if is_in_doc else "Out-Doc"
    print(f"{query[:44]:<45} | {q_type:<8} | {score:.5f}")

    if is_in_doc:
        in_doc_scores.append(score)
    else:
        out_doc_scores.append(score)

conn.close()

min_in_doc = min(in_doc_scores) if in_doc_scores else 0
max_out_doc = max(out_doc_scores) if out_doc_scores else 0

print("=" * 72)
print(f"Lowest In-Document Score : {min_in_doc:.5f}")
print(f"Highest Out-of-Doc Score  : {max_out_doc:.5f}")
print(f"Recommended Threshold     : {(min_in_doc + max_out_doc) / 2:.5f}")
