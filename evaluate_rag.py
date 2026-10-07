import psycopg2
import requests
from rank_bm25 import BM25Okapi

# 1. Connect to Database
try:
    conn = psycopg2.connect("dbname=pdf_rag user=ubuntu port=5433")
    cur = conn.cursor()
except Exception as e:
    print(f"Database connection error: {e}")
    exit(1)

def get_embedding(text):
    res = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": text})
    return res.json()["embedding"]

def tokenize(text):
    return text.lower().split()

# 2. Load BM25 Index
print("Loading BM25 index for evaluation...")
cur.execute("SELECT id, content FROM documents;")
docs = cur.fetchall()
doc_ids = [row[0] for row in docs]
doc_contents = [row[1] for row in docs]
bm25 = BM25Okapi([tokenize(c) for c in doc_contents])

# 3. Test Dataset Tailored to the Process State & PCB Document
test_dataset = [
    {
        "question": "What are the different states a process can be in according to the diagram?",
        "expected_keyword": "terminated"
    },
    {
        "question": "What does a Process Control Block store?",
        "expected_keyword": "process-specific"
    },
    {
        "question": "What does the program counter indicate in a PCB?",
        "expected_keyword": "address"
    },
    {
        "question": "What types of information are stored under CPU registers in a process control block?",
        "expected_keyword": "accumulators"
    },
    {
        "question": "What does memory-management information include?",
        "expected_keyword": "base and limit registers"
    }
]

print(f"\nRunning evaluation across {len(test_dataset)} test cases...\n")
score_hits = 0

for i, test in enumerate(test_dataset, 1):
    query = test["question"]
    keyword = test["expected_keyword"].lower()
    
    # Run Hybrid Search (Vector + BM25 + RRF)
    query_emb = get_embedding(query)
    cur.execute("SELECT id, RANK() OVER (ORDER BY embedding <=> %s::vector) as v_rank FROM documents LIMIT 10", (query_emb,))
    vector_ranks = {row[0]: row[1] for row in cur.fetchall()}

    scores = bm25.get_scores(tokenize(query))
    id_score_pairs = sorted([(doc_ids[idx], scores[idx]) for idx in range(len(doc_ids)) if scores[idx] > 0], key=lambda x: x[1], reverse=True)
    keyword_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(id_score_pairs[:10], 1)}

    rrf_scores = []
    for doc_id in set(vector_ranks.keys()).union(set(keyword_ranks.keys())):
        v_rank = vector_ranks.get(doc_id)
        k_rank = keyword_ranks.get(doc_id)
        v_score = 1.0 / (60 + v_rank) if v_rank else 0.0
        k_score = 1.0 / (60 + k_rank) if k_rank else 0.0
        rrf_scores.append((doc_id, v_score + k_score))
    
    top_3 = sorted(rrf_scores, key=lambda x: x[1], reverse=True)[:3]
    top_chunks_text = " ".join([doc_contents[doc_ids.index(doc_id)] for doc_id, _ in top_3]).lower()

    # Check if expected keyword is present in the top 3 retrieved chunks
    passed = keyword in top_chunks_text
    if passed:
        score_hits += 1
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
        
    print(f"Test {i}: {query}")
    print(f"   -> Result: {status} (Keyword '{keyword}' found in top 3 chunks: {passed})")

# Calculate Hit Rate Accuracy
accuracy = (score_hits / len(test_dataset)) * 100
print("=" * 50)
print(f"🎯 RETRIEVAL ACCURACY (Hit Rate @ 3): {accuracy:.2f}%")
print("=" * 50)

cur.close()
conn.close()
