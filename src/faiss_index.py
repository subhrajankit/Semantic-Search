import os
import json
import pickle
import faiss
import argparse
from sentence_transformers import SentenceTransformer

# ---------------------------
# Helper Functions
# ---------------------------

def load_texts(texts_path):
    """Load dataset from .pkl, .txt, or .json"""
    ext = os.path.splitext(texts_path)[1].lower()

    if ext == ".pkl":
        with open(texts_path, "rb") as f:
            return pickle.load(f)
    elif ext == ".txt":
        with open(texts_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    elif ext == ".json":
        with open(texts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                if all(isinstance(d, str) for d in data):
                    return data
                elif all(isinstance(d, dict) and "text" in d for d in data):
                    return [d["text"] for d in data]
            raise ValueError("Invalid JSON format. Must be list of strings or list of {text: ...}")
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")


def build_faiss_index(model_name, texts, index_path, embeddings_path):
    """Build and save FAISS index"""
    print(f"🔄 Building FAISS index with model: {model_name}")

    model = SentenceTransformer(model_name)

    # Encode & normalize embeddings for cosine similarity
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and embeddings
    faiss.write_index(index, index_path)
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"✅ Saved FAISS index → {index_path}")
    print(f"✅ Saved embeddings → {embeddings_path}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS Index")
    parser.add_argument("--texts", type=str, default="data/texts.pkl", help="Dataset path (.pkl, .txt, .json)")
    parser.add_argument("--index", type=str, default="data/faiss_index.bin", help="Path to save FAISS index")
    parser.add_argument("--embeddings", type=str, default="data/embeddings.pkl", help="Path to save embeddings")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = parser.parse_args()

    # Load dataset
    texts = load_texts(args.texts)
    print(f"📂 Loaded {len(texts)} texts from {args.texts}")

    # Build index
    build_faiss_index(args.model, texts, args.index, args.embeddings)
