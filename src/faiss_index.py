import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import hashlib

DATA_FILE = "data/texts.pkl"
INDEX_FILE = "data/faiss_index.bin"
EMBEDDINGS_FILE = "data/embeddings.pkl"
CHECKSUM_FILE = "data/dataset_checksum.txt"

# Function to compute checksum of dataset
def compute_checksum(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        return hashlib.md5(data).hexdigest()

# Load dataset
with open(DATA_FILE, "rb") as f:
    texts = pickle.load(f)

print(f"Loaded {len(texts)} texts.")

# Compute checksum
new_checksum = compute_checksum(DATA_FILE)

# Check if index + checksum already exist
if os.path.exists(INDEX_FILE) and os.path.exists(CHECKSUM_FILE):
    with open(CHECKSUM_FILE, "r") as f:
        old_checksum = f.read().strip()
    if old_checksum == new_checksum:
        print("✅ Dataset unchanged. Skipping FAISS index rebuild.")
        exit(0)

print("⚡ Dataset changed or index missing. Rebuilding FAISS index...")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all texts to vectors
print("Encoding texts...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)
print(f"Embeddings normalized. Shape: {embeddings.shape}")

# Create FAISS index (Inner Product for cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
print(f"FAISS index (cosine similarity) created with dimension {dimension}.")

# Add embeddings
index.add(embeddings)
print(f"Added {index.ntotal} vectors to FAISS index.")

# Save index and embeddings
faiss.write_index(index, INDEX_FILE)
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(embeddings, f)

# Save new checksum
with open(CHECKSUM_FILE, "w") as f:
    f.write(new_checksum)

print("✅ FAISS index and embeddings updated successfully.")
