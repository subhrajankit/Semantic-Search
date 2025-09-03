import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load your dataset
with open("data/texts.pkl", "rb") as f:
    texts = pickle.load(f)

print(f"Loaded {len(texts)} texts.")

# 2. Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Encode all texts to vectors
print("Encoding texts...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# 4. Normalize embeddings (so inner product = cosine similarity)
faiss.normalize_L2(embeddings)

# 5. Create FAISS index (Inner Product → cosine similarity after normalization)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
print(f"FAISS index created with dimension {dimension} using cosine similarity.")

# 6. Add embeddings to index
index.add(embeddings)
print(f"Added {index.ntotal} vectors to FAISS index.")

# 7. Save index
faiss.write_index(index, "data/faiss_index.bin")
print("FAISS index saved as 'data/faiss_index.bin'.")

# 8. Optionally save normalized embeddings too
with open("data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
print("Normalized embeddings saved as 'data/embeddings.pkl'.")
