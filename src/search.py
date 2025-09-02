# search.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load texts
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

def semantic_search(query, top_k=3):
    # Convert query to embedding
    query_vector = model.encode([query], convert_to_numpy=True)

    # Search in FAISS
    distances, indices = index.search(query_vector, top_k)

    # Collect results
    results = [(texts[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
    return results

if __name__ == "__main__":
    query = input("🔎 Enter your query: ")  # take user input
    results = semantic_search(query, top_k=3)

    print(f"\n🔍 Query: {query}\n")
    for text, dist in results:
        print(f"→ {text} (score: {dist:.4f})")
