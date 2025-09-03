import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# Paths
texts_path = "data/texts.pkl"
index_path = "data/faiss_index.bin"
embeddings_path = "data/embeddings.pkl"
meta_path = "data/index_meta.pkl"  # store dataset timestamp

# Step 1: Load dataset
if not os.path.exists(texts_path):
    raise FileNotFoundError("❌ Dataset file not found: data/texts.pkl")

with open(texts_path, "rb") as f:
    texts = pickle.load(f)

print(f"✅ Loaded {len(texts)} texts.")

# Step 2: Check if index already exists
dataset_mtime = os.path.getmtime(texts_path)

if os.path.exists(index_path) and os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    if meta.get("dataset_mtime") == dataset_mtime:
        print("⚡ FAISS index is up-to-date. Skipping rebuild.")
        exit(0)

print("🔄 Rebuilding FAISS index...")

# Step 3: Load embedding model
model_name = "all-MiniLM-L6-v2"
print(f"🔄 Using model: {model_name}")
model = SentenceTransformer(model_name)

# Step 4: Encode texts into embeddings
print("⚡ Encoding texts into embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)
print("✅ Normalized embeddings for cosine similarity.")

# Step 5: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity
print(f"✅ FAISS index created with dimension {dimension}.")

# Step 6: Add embeddings
index.add(embeddings)
print(f"✅ Added {index.ntotal} vectors to FAISS index.")

# Step 7: Save index + embeddings
faiss.write_index(index, index_path)
with open(embeddings_path, "wb") as f:
    pickle.dump(embeddings, f)

# Step 8: Save metadata (to detect dataset changes next run)
with open(meta_path, "wb") as f:
    pickle.dump({"dataset_mtime": dataset_mtime}, f)

print("💾 FAISS index and metadata saved successfully.")
