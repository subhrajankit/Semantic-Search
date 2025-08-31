from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

#example documents
documents = [
    "I love playing football.",
    "Artificial Intelligence is the future.",
    "Messi is the greatest football player.",
    "Python is a very powerful programming language.",
    "Soccer is also known as football in Europe."
]

# Encode the documents to get their embeddings
embeddings = model.encode(documents, convert_to_tensor=True)

# Example query
query = "Who is the best soccer player?"

query_embedding = model.encode(query, convert_to_tensor=True)

#Find similar documents
similarities = util.pytorch_cos_sim(query_embedding, embeddings)

for idx, score in enumerate(similarities[0]):
    print(f"Document: {documents[idx]} \nScore: {score.item():.4f}\n")

