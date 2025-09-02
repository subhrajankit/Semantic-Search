import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load your dataset
with open("data/texts.pkl", "rb") as f:
    texts = pickle.load(f)

print(f"Loaded {len(texts)} texts.")

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all texts to vectors
print("Encoding texts...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
print(f"FAISS index created with dimension {dimension}.")

# Add embeddings to index
index.add(embeddings)
print(f"Added {index.ntotal} vectors to FAISS index.")

# Save index
faiss.write_index(index, "data/faiss_index.bin")
print("FAISS index saved as 'data/faiss_index.bin'.")

# Optionally save embeddings too for reference
with open("data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)
print("Embeddings saved as 'data/embeddings.pkl'.")
