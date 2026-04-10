import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add current dir to path to import src
sys.path.append(os.getcwd())
load_dotenv()

from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import LocalEmbedder, _mock_embed
from main import load_documents_from_files, get_all_data_files

# 1. Setup with REAL model
provider = os.getenv("EMBEDDING_PROVIDER", "mock")
if provider == "local":
    embedder = LocalEmbedder()
else:
    embedder = _mock_embed

print(f"Using Embedder: {embedder.__class__.__name__}")

all_files = get_all_data_files()
docs = load_documents_from_files(all_files)
store = EmbeddingStore(embedding_fn=embedder)
store.add_documents(docs)

def mock_llm(prompt):
    return f"[Local RAG Output] Based on real context..."

agent = KnowledgeBaseAgent(store, llm_fn=mock_llm)

# 2. Queries from benchmark_queries.md
queries = [
    "In NIPT, what is the role of paternal DNA information?",
    "What genetic factor determines the subtype (severity category) of alpha-thalassemia?",
    "What is the basic human chromosome makeup described here?",
    "What is the most common malignant brain tumour in children?",
    "Why can brain tumours cause headaches and seizures?"
]

print("| # | Query | Top-1 Doc | Score | Answer Preview |")
print("|---|-------|-----------|-------|----------------|")

for i, q in enumerate(queries, 1):
    results = store.search(q, top_k=1)
    top_doc = results[0] if results else {"metadata": {"source": "None"}, "score": 0, "content": "N/A"}
    source = Path(top_doc['metadata']['source']).name
    score = f"{top_doc['score']:.4f}"
    
    # Run agent
    ans = agent.answer(q, top_k=1)
    
    print(f"| {i} | {q} | {source} | {score} | {ans[:60]}... |")
