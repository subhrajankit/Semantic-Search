import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("data/faiss_index.bin")

# Load original text data
with open("data/sentences.pkl", "rb") as f:
    sentences = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query, top_k=3):
    # Convert query to vector
    query_vector = model.encode([query])

    # Ensure FAISS gets correct dimensions
    query_vector = query_vector.reshape(1, -1)

    # Search the FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Get the matched sentences
    results = [sentences[i] for i in indices[0]]
    return results

if __name__ == "__main__":
    print("🔍 Semantic Search Engine (type 'exit' to quit)")
    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            print("Exiting search...")
            break
        results = semantic_search(query)
        print("\nTop Results:")
        for r in results:
            print("-", r)
