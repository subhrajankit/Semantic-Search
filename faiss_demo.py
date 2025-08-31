from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Documents
documents = [
    "I love playing football.",
    "Artificial Intelligence is the future.",
    "Messi is the greatest football player.",
    "Python is a very powerful programming language.",
    "Soccer is also known as football in Europe."
]

# Step 3: Convert documents to embeddings
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Step 4: Initialize FAISS index (using cosine similarity)
dimension = doc_embeddings.shape[1]  # Embedding size
index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine similarity after normalization)

# Normalize vectors (for cosine similarity)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

print("Number of documents in index:", index.ntotal)

# Step 5: Query
query = "Who is the best soccer player?"
query_embedding = model.encode([query], convert_to_numpy=True)
faiss.normalize_L2(query_embedding)

# Step 6: Search top 3 results
k = 3
distances, indices = index.search(query_embedding, k)

print("\nQuery:", query)
print("\nTop results:")
for i, idx in enumerate(indices[0]):
    print(f"{documents[idx]} (Score: {distances[0][i]:.4f})")
