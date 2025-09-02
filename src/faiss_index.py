# faiss_index.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample dataset (replace/add your own)
texts = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning is a subset of machine learning.",
    "Python is a popular programming language.",
    "FAISS is a library for efficient similarity search.",
    "Neural networks can approximate complex functions."
]

# Convert texts into embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Create FAISS index
d = embeddings.shape[1]  # dimension of embeddings
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.bin")

# Save texts for later retrieval
with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ FAISS index and texts saved successfully!")
