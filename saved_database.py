# save_dataset.py
import pickle

texts = [
    # Technology
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks for solving complex tasks.",
    "Python is one of the most popular programming languages.",
    "FAISS is a library for efficient similarity search.",
    "Cloud computing enables scalable data storage and computing power.",

    # History
    "The Indus Valley Civilization is one of the oldest in the world.",
    "The Mughal Empire ruled most of India for several centuries.",
    "Mahatma Gandhi led India’s struggle for independence.",
    "World War II ended in 1945 after the surrender of Japan.",
    "The French Revolution began in 1789.",

    # Products / E-commerce
    "Flipkart is one of India’s leading e-commerce platforms.",
    "Amazon started as an online bookstore before expanding.",
    "iPhone 15 is Apple’s latest flagship smartphone.",
    "Electric vehicles are becoming increasingly popular.",
    "The PlayStation 5 offers powerful gaming performance.",

    # Quotes
    "The only limit to our realization of tomorrow is our doubts of today.",
    "In the middle of every difficulty lies opportunity.",
    "Success is not final, failure is not fatal: it is the courage to continue that counts.",
    "Do what you can with what you have where you are.",
    "The best way to predict the future is to invent it.",

    # Random Knowledge
    "Water boils at 100 degrees Celsius at standard pressure.",
    "Mount Everest is the highest mountain in the world.",
    "The human brain contains around 86 billion neurons.",
    "Photosynthesis is the process by which plants make food.",
    "The Earth revolves around the Sun once every 365 days."
]

# Save dataset
with open("data/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print(f"Saved {len(texts)} texts to data/texts.pkl")
