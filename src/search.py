import os
import json
import pickle
import faiss
import argparse
import logging
from sentence_transformers import SentenceTransformer

# ---------------------------
# Setup Logging
# ---------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------
# Helper Functions
# ---------------------------

DATASET_PATH = "data/texts.pkl"
INDEX_PATH = "data/faiss_index.bin"
EMBEDDINGS_PATH = "data/embeddings.pkl"


def load_config():
    """Load model name from config.json"""
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            return json.load(f)
    return {"model": "all-MiniLM-L6-v2"}  # fallback default


def load_texts(path=DATASET_PATH):
    """Load dataset from pickle"""
    if not os.path.exists(path):
        logging.warning(f"Dataset file not found: {path}")
        return []
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return []


def save_texts(texts, path=DATASET_PATH):
    """Save dataset to pickle"""
    try:
        with open(path, "wb") as f:
            pickle.dump(texts, f)
        logging.info(f"Saved {len(texts)} texts to {path}")
    except Exception as e:
        logging.error(f"Failed to save dataset: {e}")


def save_embeddings_and_index(model_name, texts):
    """Build FAISS index from texts and save"""
    logging.info(f"Rebuilding FAISS index using model: {model_name}")

    model = SentenceTransformer(model_name)

    # Encode & normalize embeddings
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    try:
        faiss.write_index(index, INDEX_PATH)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)
        logging.info(f"Index saved at {INDEX_PATH}")
    except Exception as e:
        logging.error(f"Failed to save FAISS index: {e}")

    return index, model


def load_index_and_model(model_name, rebuild=False):
    """Load FAISS index and model, rebuild if missing/outdated"""
    texts = load_texts(DATASET_PATH)
    if not texts:
        raise ValueError("❌ Dataset is empty. Add entries first using `add` command.")

    if rebuild or not os.path.exists(INDEX_PATH):
        logging.warning("FAISS index not found or rebuild forced.")
        return save_embeddings_and_index(model_name, texts), texts

    try:
        index = faiss.read_index(INDEX_PATH)
        model = SentenceTransformer(model_name)
        logging.info("Loaded existing FAISS index.")
        return (index, model), texts
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}, rebuilding...")
        return save_embeddings_and_index(model_name, texts), texts


def semantic_search(query, index, model, texts, top_k=5, threshold=0.5):
    """Perform semantic search"""
    if not query.strip():
        logging.warning("Empty query received.")
        return []

    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    try:
        distances, indices = index.search(query_vector, top_k)
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return []

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(texts):  # avoid index errors
            score = distances[0][i]
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
            print(json.dumps([{"rank": i + 1, "score": s, "text": t}
                              for i, (s, t) in enumerate(results)], indent=2))
    else:
        print("⚠️ No relevant results found.")


# ---------------------------
# Interactive Mode
# ---------------------------
def interactive_mode(model_name):
    try:
        (index, model), texts = load_index_and_model(model_name)
    except Exception as e:
        logging.error(f"Failed to start interactive mode: {e}")
        print("❌ Cannot start interactive mode. Check logs for details.")
        return

    print("\n💡 Interactive mode started.")
    print("Type: search <query>, add <text>, remove <index>, list, exit\n")

    while True:
        try:
            user_input = input(">> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting interactive mode.")
                logging.info("Interactive mode exited by user.")
                break

            if user_input.startswith("search "):
                query = user_input.replace("search ", "", 1)
                results = semantic_search(query, index, model, texts, 5, 0.5)
                display_results(query, results, "table")

            elif user_input.startswith("add "):
                new_text = user_input.replace("add ", "", 1)
                texts.append(new_text)
                save_texts(texts)
                index, model = save_embeddings_and_index(model_name, texts)
                print(f"✅ Added: {new_text}")
                logging.info(f"Added new text: {new_text}")

            elif user_input.startswith("remove "):
                try:
                    idx = int(user_input.replace("remove ", "", 1))
                    if 1 <= idx <= len(texts):
                        removed = texts.pop(idx - 1)
                        save_texts(texts)
                        if texts:
                            index, model = save_embeddings_and_index(model_name, texts)
                        else:
                            if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
                            if os.path.exists(EMBEDDINGS_PATH): os.remove(EMBEDDINGS_PATH)
                        print(f"🗑 Removed: {removed}")
                        logging.info(f"Removed entry: {removed}")
                    else:
                        print("❌ Invalid index.")
                        logging.warning("Invalid remove index used.")
                except ValueError:
                    print("⚠️ Please provide a valid number.")
                    logging.warning("Non-numeric index provided for remove.")

            elif user_input == "list":
                if not texts:
                    print("📂 Dataset is empty.")
                else:
                    print("📂 Dataset entries:")
                    for i, t in enumerate(texts, 1):
                        print(f"{i}. {t}")

            else:
                print("⚠️ Unknown command. Use: search/add/remove/list/exit")
        except Exception as e:
            logging.error(f"Error in interactive loop: {e}")
            print("⚠️ An error occurred. Check logs for details.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search Engine with dataset management")
    subparsers = parser.add_subparsers(dest="command")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search query in FAISS index")
    search_parser.add_argument("--query", type=str, required=True, help="Search query text")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    search_parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (0-1)")
    search_parser.add_argument("--format", type=str, default="table", help="Output format: table or json")
    search_parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS index")

    # Dataset commands
    subparsers.add_parser("list", help="List all entries in the dataset")

    add_parser = subparsers.add_parser("add", help="Add new entry to dataset")
    add_parser.add_argument("--text", type=str, required=True, help="Text to add")

    remove_parser = subparsers.add_parser("remove", help="Remove entry by index")
    remove_parser.add_argument("--index", type=int, required=True, help="Index of entry to remove (1-based)")

    subparsers.add_parser("clear", help="Clear all entries from dataset")

    args = parser.parse_args()

    # Load config (default model)
    config = load_config()
    model_name = config["model"]

    # If no command → interactive mode
    if args.command is None:
        interactive_mode(model_name)

    # Handle commands
    elif args.command == "search":
        (index, model), texts = load_index_and_model(model_name, rebuild=args.rebuild)
        results = semantic_search(args.query, index, model, texts, args.top_k, args.threshold)
        display_results(args.query, results, output_format=args.format)

    elif args.command == "list":
        texts = load_texts()
        if not texts:
            print("📂 Dataset is empty.")
        else:
            print("📂 Dataset entries:")
            for i, t in enumerate(texts, 1):
                print(f"{i}. {t}")

    elif args.command == "add":
        texts = load_texts()
        texts.append(args.text)
        save_texts(texts)
        logging.info(f"Added via CLI: {args.text}")
        print(f"✅ Added new entry. Dataset now has {len(texts)} items.")
        save_embeddings_and_index(model_name, texts)

    elif args.command == "remove":
        texts = load_texts()
        if 1 <= args.index <= len(texts):
            removed = texts.pop(args.index - 1)
            save_texts(texts)
            logging.info(f"Removed via CLI: {removed}")
            print(f"🗑 Removed: {removed}")
            if texts:
                save_embeddings_and_index(model_name, texts)
            else:
                if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
                if os.path.exists(EMBEDDINGS_PATH): os.remove(EMBEDDINGS_PATH)
        else:
            print("❌ Invalid index.")

    elif args.command == "clear":
        save_texts([])
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
        logging.info("Dataset cleared via CLI.")
        print("⚠️ Cleared dataset and deleted FAISS index.")
