import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import json
import os
import subprocess
import hashlib

DATA_FILE = "data/texts.pkl"
INDEX_FILE = "data/faiss_index.bin"
CHECKSUM_FILE = "data/dataset_checksum.txt"

# Function to compute dataset checksum
def compute_checksum(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()

# Ensure FAISS index is up to date
def ensure_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHECKSUM_FILE):
        print("⚡ Index missing. Rebuilding...")
        subprocess.run(["python", "src/faiss_index.py"])
        return

    new_checksum = compute_checksum(DATA_FILE)
    with open(CHECKSUM_FILE, "r") as f:
        old_checksum = f.read().strip()

    if old_checksum != new_checksum:
        print("⚡ Dataset changed. Rebuilding FAISS index...")
        subprocess.run(["python", "src/faiss_index.py"])
    else:
        print("✅ FAISS index is up to date.")

# Run index check
ensure_index()

# Load FAISS index and dataset
index = faiss.read_index(INDEX_FILE)
with open(DATA_FILE, "rb") as f:
    texts = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=5, threshold=0.5):
    # Encode query
    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)  # normalize query too!

    # Search in FAISS index
    similarities, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        score = similarities[0][i]  # already cosine similarity [0,1]
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
            print(json.dumps(
                [{"rank": i+1, "score": float(s), "text": t} for i, (s, t) in enumerate(results)],
                indent=2
            ))
        else:
            print("⚠️ Invalid format. Showing table by default.")
            for rank, (score, text) in enumerate(results, 1):
                print(f"{rank}. ({score:.2f}) {text}")
    else:
        print("No relevant results found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search Engine")
    parser.add_argument("--query", type=str, required=True, help="Search query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (0-1)")
    parser.add_argument("--format", type=str, default="table", help="Output format: table or json")
    args = parser.parse_args()

    results = semantic_search(args.query, top_k=args.top_k, threshold=args.threshold)
    display_results(args.query, results, output_format=args.format)
