import os
import json
import pickle
import faiss
import numpy as np
import argparse
import logging
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ---------------------------
# Logging Setup
# ---------------------------
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/semantic_search.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ---------------------------
# Helper Functions
# ---------------------------

def load_config():
    """Load model name from config.json"""
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            return json.load(f)
    return {"model": "all-MiniLM-L6-v2"}  # fallback default


def save_embeddings_and_index(model_name, texts, index_path, embeddings_path):
    """Build FAISS index from texts and save"""
    logging.info(f"Rebuilding FAISS index using model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode & normalize embeddings for cosine similarity
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    # Create index (Inner Product since we normalized)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and embeddings
    faiss.write_index(index, index_path)
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)

    logging.info(f"Index rebuilt and saved at {index_path}")
    return index, model


def load_index_and_model(model_name, index_path="data/faiss_index.bin", texts_path="data/texts.pkl"):
    """Load FAISS index and model, rebuild if missing/outdated"""
    if not os.path.exists(texts_path):
        logging.error("Dataset (texts.pkl) not found in data/")
        raise FileNotFoundError("❌ Dataset (texts.pkl) not found in data/")

    # Load texts
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    if not os.path.exists(index_path):
        logging.warning("FAISS index not found. Rebuilding...")
        return save_embeddings_and_index(model_name, texts, index_path, "data/embeddings.pkl"), texts

    index = faiss.read_index(index_path)
    model = SentenceTransformer(model_name)
    logging.info("Loaded existing FAISS index successfully.")
    return (index, model), texts


def semantic_search(query, index, model, texts, top_k=5, threshold=0.5):
    """Perform semantic search"""
    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        score = distances[0][i]  # cosine similarity after normalization
        if score >= threshold:
            results.append((score, texts[idx]))

    logging.info(f"Query '{query}' returned {len(results)} results.")
    return results


def display_results(query, results, output_format="table"):
    """Display results in table or JSON format"""
    print(f"\n🔍 Query: {query}\n")
    if results:
        if output_format == "table":
            print("Top Results:")
            for rank, (score, text) in enumerate(results, 1):
                print(f"{rank}. ({score:.2f}) {text}")
        elif output_format == "json":
            import json
            print(json.dumps([{"rank": i+1, "score": s, "text": t}
                              for i, (s, t) in enumerate(results)], indent=2))
    else:
        print("⚠️ No relevant results found.")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search Engine")
    parser.add_argument("--query", type=str, help="Search query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (0-1)")
    parser.add_argument("--format", type=str, default="table", help="Output format: table or json")
    parser.add_argument("--model", type=str, help="Override embedding model")
    args = parser.parse_args()

    # Load config (default model)
    config = load_config()
    model_name = args.model if args.model else config["model"]

    # Load or rebuild index
    (index, model), texts = load_index_and_model(model_name)

    if args.query:  # CLI mode
        results = semantic_search(args.query, index, model, texts, args.top_k, args.threshold)
        display_results(args.query, results, output_format=args.format)
    else:  # Interactive mode
        print("💡 Enter your query (type 'exit' to quit):")
        while True:
            query = input("\nQuery: ")
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting search.")
                logging.info("User exited interactive search session.")
                break
            results = semantic_search(query, index, model, texts, args.top_k, args.threshold)
            display_results(query, results, output_format=args.format)
