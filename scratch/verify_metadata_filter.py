import sys
import os
from pathlib import Path

# Add current dir to path to import src
sys.path.append(os.getcwd())

from src.store import EmbeddingStore
from src.embeddings import _mock_embed
from main import load_documents_from_files, get_all_data_files

# 1. Setup
docs = load_documents_from_files(get_all_data_files())
store = EmbeddingStore(embedding_fn=_mock_embed)
store.add_documents(docs)

# 2. Test Search with Metadata Filtering
# Query "headaches" but filter for "Brain tumor" category
print("=== Testing Metadata Filtering ===")
query = "symptoms and headaches"
category_filter = "Brain tumor"

results = store.search_with_filter(
    query, 
    metadata_filter={"category": category_filter}, 
    top_k=2
)

print(f"Results for '{query}' with category='{category_filter}':")
for r in results:
    source = r['metadata'].get('source', 'N/A')
    cat = r['metadata'].get('category', 'N/A')
    print(f"- Source: {source}")
    print(f"  Category: {cat}")
    print(f"  Score: {r['score']:.4f}")
    print(f"  Content snippet: {r['content'][:100]}...")
    print("-" * 20)
