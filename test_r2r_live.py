import time
from r2r import R2RClient
from shared.abstractions.exception import R2RException

client = R2RClient(base_url="http://localhost:7272")

# 1. Check server health
print("1. Health:", client.system.health())

# 2. Ingest only if not already present
file_name = "company_policy.md"
existing_docs = client.documents.list().get("results", [])
already_ingested = any(doc.get("title") == file_name for doc in existing_docs)

if already_ingested:
    print(f"\n2. '{file_name}' is already ingested. Skipping ingestion.")
else:
    print(f"\n2. Ingesting {file_name}...")
    try:
        ingest_res = client.documents.create(file_path=file_name)
        print("Ingest response:", ingest_res)
        time.sleep(5)
    except R2RException as e:
        if "already exists" in str(e):
            print(f"'{file_name}' already exists in database. Continuing...")
        else:
            raise

# 3. Query the RAG pipeline
query = "What is the refund window for Enterprise Plan customers, and what is the P1 SLA?"
print(f"\n3. Querying RAG: '{query}'")
rag_res = client.retrieval.rag(
    query=query,
    search_settings={
        "use_hybrid_search": False,
        "graph_search_settings": {"enabled": False},
        }
)

results = rag_res.get("results", {})
print("\n[DEBUG KEYS]:", list(results.keys()))

print("\n--- GENERATED ANSWER ---")
print(results.get("completion") or results.get("generated_answer") or "")

print("\n--- RETRIEVED CHUNKS ---")
chunks = results.get("search_results", {}).get("chunk_search_results", [])
for i, c in enumerate(chunks, 1):
    score = c.get("score", 0.0)
    text = c.get("text", "")
    print(f"[Chunk {i} | Score: {score:.3f}]: {text[:150]}...")

print("\n--- CITATIONS ---")
print(results.get("citations", []))
