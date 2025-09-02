import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load FAISS index
index = faiss.read_index("data/faiss_index.bin")

# Load original texts
with open("data/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5, output_format="table"):
    # Encode query into vector
    query_vector = model.encode([query]).astype(np.float32)

    # Search in FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Collect results
    results = []
    for i, idx in enumerate(indices[0]):
        score = 1 - distances[0][i]  # similarity score
        results.append({
            "rank": i + 1,
            "score": round(float(score), 3),
            "text": texts[idx]
        })

    # Print query and results
    print(f"\n🔎 Query: {query}\nTop {top_k} Results:\n")

    if output_format == "table":
        for r in results:
            print(f"{r['rank']}. ({r['score']}) {r['text']}")
    elif output_format == "json":
        print(json.dumps(results, indent=2))
    else:
        print("⚠️ Invalid format! Defaulting to table.")
        for r in results:
            print(f"{r['rank']}. ({r['score']}) {r['text']}")

if __name__ == "__main__":
    query = input("Enter your search query: ")
    output_format = input("Choose output format (table/json): ").strip().lower()
    semantic_search(query, top_k=5, output_format=output_format)
