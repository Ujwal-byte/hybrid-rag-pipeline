import streamlit as st
import psycopg2
import requests
conn = psycopg2.connect("dbname=pdf_rag user=ubuntu port=5433")
st.title("PDF Hybrid RAG Pipeline")
query = st.text_input("Ask a question about your PDF documents:")

if st.button("Search & Answer") and query:
    with st.spinner("Executing Hybrid Search and generating answer..."):
        # 1. Get embedding for the user query
        query_emb = requests.post("http://localhost:11434/api/embeddings", 
                                  json={"model": "nomic-embed-text", "prompt": query}).json()["embedding"]

        # 2. Reciprocal Rank Fusion (RRF) SQL Query combining Dense Vector + Sparse Keyword search
        cur = conn.cursor()
        rrf_sql = """
        WITH vector_search AS (
            SELECT id, content, RANK() OVER (ORDER BY embedding <=> %s::vector) AS rank
            FROM documents ORDER BY embedding <=> %s::vector LIMIT 10
        ),
        keyword_search AS (
            SELECT id, content, RANK() OVER (ORDER BY ts_rank(fts_tokens, plainto_tsquery('english', %s)) DESC) AS rank
            FROM documents WHERE fts_tokens @@ plainto_tsquery('english', %s) LIMIT 10
        )
        SELECT COALESCE(v.content, k.content)
        FROM vector_search v FULL OUTER JOIN keyword_search k ON v.id = k.id
        ORDER BY (COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) DESC
        LIMIT 3;
        """
        cur.execute(rrf_sql, (query_emb, query_emb, query, query))
        results = cur.fetchall()
        context = "\n\n".join([r[0] for r in results])

        # 3. Generate response with Llama 3.2
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nProvide a concise and accurate answer based on the context:"
        response = requests.post("http://localhost:11434/api/generate", 
                                 json={"model": "llama3.2", "prompt": prompt, "stream": False})

        st.write("### Answer")
        st.write(response.json()["response"])

        with st.expander("Show Retrieved Context Chunks"):
            st.write(context)
