import faiss
import pickle

# Save FAISS index
def save_index(index, file_path="data/faiss_index.bin"):
    faiss.write_index(index, file_path)
    print(f"[INFO] Index saved to {file_path}")

# Load FAISS index
def load_index(file_path="data/faiss_index.bin"):
    index = faiss.read_index(file_path)
    print(f"[INFO] Index loaded from {file_path}")
    return index

# Save Python objects (e.g., texts, sentences)
def save_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Pickle saved to {file_path}")

# Load Python objects
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    print(f"[INFO] Pickle loaded from {file_path}")
    return obj
