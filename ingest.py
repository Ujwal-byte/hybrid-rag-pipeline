import sys
import fitz  # PyMuPDF
import psycopg2
import requests

if len(sys.argv) < 2:
    print("Error: Please specify a PDF file. Example: python ingest.py module1.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Starting ingestion for file: {pdf_path}")

try:
    conn = psycopg2.connect("dbname=pdf_rag user=ubuntu port=5433")
    cur = conn.cursor()
    print("Successfully connected to ShaktiDB on port 5433.")
except Exception as e:
    print(f"Database connection error: {e}")
    sys.exit(1)

def get_embedding(text):
    res = requests.post(
        "http://localhost:11434/api/embeddings", 
        json={"model": "nomic-embed-text", "prompt": text}
    )
    if res.status_code != 200:
        raise Exception(f"Ollama embedding error: {res.text}")
    return res.json()["embedding"]

try:
    doc = fitz.open(pdf_path)
    print(f"PDF opened successfully. Total pages: {len(doc)}")
except Exception as e:
    print(f"Failed to open PDF file '{pdf_path}': {e}")
    sys.exit(1)

inserted_count = 0

# Helper function to create sliding character chunks
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

for page_num in range(len(doc)):
    page = doc[page_num]
    page_text = page.get_text().strip()
    
    # Fallback to OCR if page has no digital text
    if not page_text:
        print(f"Running OCR on Page {page_num + 1}...")
        try:
            tp = page.get_textpage_ocr(language="eng")
            page_text = page.get_text(textpage=tp).strip()
        except Exception as ocr_err:
            print(f"OCR failed on Page {page_num + 1}: {ocr_err}")

    if not page_text:
        print(f"Skipping Page {page_num + 1}: No readable text found.")
        continue

    # Split page text into optimized sliding chunks
    text_chunks = chunk_text(page_text, chunk_size=500, overlap=100)
    print(f"Processing Page {page_num + 1}: Generated {len(text_chunks)} chunks...")

    for chunk in text_chunks:
        if not chunk.strip():
            continue
        try:
            embedding = get_embedding(chunk)
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (chunk, embedding)
            )
            inserted_count += 1
        except Exception as e:
            print(f"Error inserting chunk: {e}")

conn.commit()
cur.close()
conn.close()

print("=" * 60)
print(f"Ingestion complete! Successfully inserted {inserted_count} chunks into 'pdf_rag'.")
print("=" * 60)
