import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import json

# Load FAISS index and texts
index = faiss.read_index("data/faiss_index.bin")
with open("data/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5, threshold=0.5):
    # Encode query
    query_vector = model.encode([query]).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(query_vector)

    # Search
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        score = float(distances[0][i])  # cosine similarity score
        if score >= threshold:
            results.append((score, texts[idx]))

    return results

def display_results(query, results, output_format="table"):
    print(f"\nQuery: {query}\n")
    if results:
        if output_format == "table":
            print("Top Results:")
            for rank, (score, text) in enumerate(results, 1):
                print(f"{rank}. ({score:.2f}) {text}")
        elif output_format == "json":
            print(json.dumps([{"rank": i+1, "score": s, "text": t} 
                              for i, (s, t) in enumerate(results)], indent=2))
        else:
            print("⚠️ Invalid format. Showing table by default.")
            for rank, (score, text) in enumerate(results, 1):
                print(f"{rank}. ({score:.2f}) {text}")
    else:
        print("No relevant results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search Engine")
    parser.add_argument("--query", type=str, help="Search query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (0-1)")
    parser.add_argument("--format", type=str, default="table", help="Output format: table or json")
    args = parser.parse_args()

    if args.query:  
        # ✅ CLI Mode
        results = semantic_search(args.query, top_k=args.top_k, threshold=args.threshold)
        display_results(args.query, results, output_format=args.format)
    else:  
        # ✅ Interactive Mode
        print("Interactive Semantic Search (type 'exit' to quit)\n")
        while True:
            query = input("Enter your query: ")
            if query.lower() == "exit":
                break
            results = semantic_search(query, top_k=args.top_k, threshold=args.threshold)
            display_results(query, results, output_format=args.format)
