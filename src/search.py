import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index and texts
index = faiss.read_index("data/faiss_index.bin")
with open("data/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5, threshold=0.5):
    # Encode query
    query_vector = model.encode([query]).astype("float32")

    # Search in FAISS index
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        # Convert distance to similarity (approximate)
        score = 1 - distances[0][i] / 2
        if score >= threshold:  # apply filtering
            results.append((score, texts[idx]))

    return results


if __name__ == "__main__":
    while True:
        query = input("\nEnter your search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting search. Goodbye 👋")
            break

        results = semantic_search(query, top_k=5, threshold=0.5)

        print(f"\nQuery: {query}\n")
        if results:
            print("Top Results:")
            for rank, (score, text) in enumerate(results, 1):
                print(f"{rank}. ({score:.2f}) {text}")
        else:
            print("No relevant results found.")
