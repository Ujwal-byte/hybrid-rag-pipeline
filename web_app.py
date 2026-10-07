import streamlit as st
import fitz  # PyMuPDF
import psycopg2
import requests
import json
import time
import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- NLTK Data Pre-loading ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# --- Page Config ---
st.set_page_config(page_title="Transparent Hybrid RAG Chat", layout="centered")
st.title("📚 Transparent Hybrid RAG (Semantic Chunking+ Semantic caching with conversational memeory and Reranking with cross-encoder)")

# --- Semantic Chunking Function ---
def semantic_chunk_text(text, max_chunk_size=800, overlap_sentences=1):
    """Chunks text cleanly along sentence boundaries rather than character cuts."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlapping sentences for context continuity
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences < len(current_chunk) else current_chunk
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_embedding(text):
    res = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": text})
    return res.json()["embedding"]

def tokenize(text):
    return text.lower().split()

def rewrite_query(current_query, chat_history):
    if not chat_history:
        return current_query

    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]])
    prompt = f"""Given the following conversation history and the user's latest query, rewrite the query to be a standalone question that captures all necessary context from the history.
If the query is already standalone, just return the original query. 
Respond with ONLY the standalone query text, without quotes, explanations, or conversational filler.

Conversation History:
{history_text}

Latest Query: {current_query}

Standalone Query:"""

    res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False})
    return res.json()["response"].strip().strip('"').strip("'")

# --- Load Models & Database ---
@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def init_db():
    conn = psycopg2.connect("dbname=pdf_rag user=ubuntu port=5433")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id SERIAL PRIMARY KEY,
            query_text TEXT,
            query_embedding vector(768),
            llm_answer TEXT
        );
    """)
    conn.commit()
    return conn

cross_encoder = load_cross_encoder()
conn = init_db()
cur = conn.cursor()

# --- Sidebar: Direct PDF File Ingestion ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process & Ingest PDF"):
            with st.spinner("Semantic sentence chunking, OCR scanning, and vectorizing..."):
                try:
                    cur.execute("TRUNCATE TABLE documents; TRUNCATE TABLE query_cache;")
                    conn.commit()

                    pdf_bytes = uploaded_file.getvalue()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        page_text = page.get_text().strip()

                        try:
                            tp = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                            ocr_text = page.get_text(textpage=tp).strip()
                            if len(ocr_text) > len(page_text):
                                page_text = ocr_text
                        except Exception:
                            pass

                        # Apply NLTK Semantic Chunking
                        text_chunks = semantic_chunk_text(page_text, max_chunk_size=800, overlap_sentences=1)
                        for chunk in text_chunks:
                            if chunk.strip():
                                cur.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                                            (chunk, get_embedding(chunk)))

                    conn.commit()
                    st.cache_data.clear()
                    st.success("PDF semantically ingested!")
                    st.session_state.messages = []

                except Exception as e:
                    st.error(f"Error processing uploaded PDF: {e}")

# --- Load BM25 Index into Memory ---
@st.cache_data
def load_bm25_index():
    cur.execute("SELECT id, content FROM documents;")
    docs = cur.fetchall()
    if not docs: return None, None, None
    doc_ids = [row[0] for row in docs]
    doc_contents = [row[1] for row in docs]
    bm25 = BM25Okapi([tokenize(c) for c in doc_contents])
    return bm25, doc_ids, doc_contents

bm25, doc_ids, doc_contents = load_bm25_index()

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "diagnostics" in message:
            with st.expander("🔍 RAG Pipeline Internal Diagnostics"):
                st.markdown(message["diagnostics"])

if query := st.chat_input("Ask a question about your document..."):
    if not bm25 or not doc_contents:
        st.error("Please upload and ingest a PDF from the sidebar first!")
    else:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing conversation history..."):
                standalone_query = rewrite_query(query, st.session_state.messages)

            st.session_state.messages.append({"role": "user", "content": query})
            query_emb = get_embedding(standalone_query)

            # --- SEMANTIC CACHE CHECK ---
            cur.execute("""
                SELECT llm_answer, 1 - (query_embedding <=> %s::vector) AS similarity 
                FROM query_cache 
                ORDER BY query_embedding <=> %s::vector 
                LIMIT 1;
            """, (query_emb, query_emb))

            cache_match = cur.fetchone()

            if cache_match and cache_match[1] >= 0.95:
                cached_answer = cache_match[0]
                similarity_score = cache_match[1]

                def stream_cache():
                    for word in cached_answer.split(" "):
                        yield word + " "
                        time.sleep(0.01)

                full_answer = st.write_stream(stream_cache())

                diag_md = f"""⚡ **SEMANTIC CACHE HIT!**\n
* **Rewritten Query:** `{standalone_query}`
* **Similarity Score:** `{similarity_score:.5f}`"""

                with st.expander("🔍 RAG Pipeline Internal Diagnostics"):
                    st.markdown(diag_md)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_answer,
                    "diagnostics": diag_md
                })

            # --- CACHE MISS (Run Full Hybrid + Reranker) ---
            else:
                with st.spinner("Running hybrid search & cross-encoder reranking..."):
                    # Vector & BM25 Search
                    cur.execute("SELECT id, RANK() OVER (ORDER BY embedding <=> %s::vector) as v_rank FROM documents LIMIT 10", (query_emb,))
                    vector_ranks = {row[0]: row[1] for row in cur.fetchall()}

                    query_tokens = tokenize(standalone_query)
                    scores = bm25.get_scores(query_tokens)
                    id_score_pairs = sorted([(doc_ids[i], scores[i]) for i in range(len(doc_ids)) if scores[i] > 0], key=lambda x: x[1], reverse=True)
                    keyword_ranks = {doc_id: rank for rank, (doc_id, score) in enumerate(id_score_pairs[:10], 1)}

                    # RRF Scoring
                    rrf_scores = []
                    for doc_id in set(vector_ranks.keys()).union(set(keyword_ranks.keys())):
                        v_rank = vector_ranks.get(doc_id)
                        k_rank = keyword_ranks.get(doc_id)
                        v_score = 1.0 / (60 + v_rank) if v_rank else 0.0
                        k_score = 1.0 / (60 + k_rank) if k_rank else 0.0
                        rrf_scores.append((doc_id, v_score + k_score))

                    top_10_rrf = sorted(rrf_scores, key=lambda x: x[1], reverse=True)[:10]
                    candidate_chunks = [(doc_id, doc_contents[doc_ids.index(doc_id)]) for doc_id, _ in top_10_rrf]

                    # Cross-Encoder Reranking
                    cross_inp = [[standalone_query, chunk_text] for _, chunk_text in candidate_chunks]
                    ce_scores = cross_encoder.predict(cross_inp)

                    # Select Top 3 semantic chunks (High precision, lower prompt length for fast CPU generation)
                    reranked_results = sorted(zip(candidate_chunks, ce_scores), key=lambda x: x[1], reverse=True)[:3]
                    context_chunks = [item[0][1] for item in reranked_results]

                    # LLM Generation
                    context = "\n\n".join(context_chunks)
                    prompt = f"Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context, answer the query: {standalone_query}"

                    gen_res = requests.post(
                        "http://localhost:11434/api/generate", 
                        json={"model": "llama3.2", "prompt": prompt, "stream": True}, 
                        stream=True
                    )

                    def stream_tokens():
                        for line in gen_res.iter_lines():
                            if line:
                                yield json.loads(line)["response"]

                    full_answer = st.write_stream(stream_tokens())

                    cur.execute("INSERT INTO query_cache (query_text, query_embedding, llm_answer) VALUES (%s, %s, %s)", 
                                (standalone_query, query_emb, full_answer))
                    conn.commit()

                    diag_md = f"""**0. Strategy:** Semantic Sentence Chunking + Top 3 Cross-Encoder Reranking\n
* Rewritten Query: `{standalone_query}`

**1. Top 3 Clean Context Chunks Fed to Llama 3.2:**"""

                    for idx, (chunk_data, ce_score) in enumerate(reranked_results, 1):
                        doc_id = chunk_data[0]
                        chunk_text = chunk_data[1]
                        original_rrf_rank = next(i for i, v in enumerate(top_10_rrf) if v[0] == doc_id) + 1
                        diag_md += f"\n* **Rank {idx}** (Was RRF Rank {original_rrf_rank} | CE Score: `{ce_score:.3f}`)\n  > *{chunk_text}*\n"

                    with st.expander("🔍 RAG Pipeline Internal Diagnostics"):
                        st.markdown(diag_md)

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_answer,
                        "diagnostics": diag_md
                    })
