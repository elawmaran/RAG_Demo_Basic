import os
import pickle
import numpy as np
import faiss

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 1. Load OpenAI API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in .env")

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# 2. Load saved chunks
# --------------------------------------------------

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]


# --------------------------------------------------
# 3. Load FAISS index
# --------------------------------------------------

index = faiss.read_index("faiss.index")

print(f"FAISS index loaded.")
print(f"Number of vectors: {index.ntotal}")


# --------------------------------------------------
# 4. Ask a question
# --------------------------------------------------

question = input("\nEnter your question: ")

print("\nCreating embedding for your question...")


# --------------------------------------------------
# 5. Convert question into an embedding
# --------------------------------------------------

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

question_embedding = response.data[0].embedding


# --------------------------------------------------
# 6. Convert to NumPy array
# --------------------------------------------------

query_vector = np.array(
    [question_embedding],
    dtype="float32"
)


# --------------------------------------------------
# 7. Search FAISS
# --------------------------------------------------

k = 3

distances, indices = index.search(
    query_vector,
    k
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\nTop 3 relevant chunks:")
print("=" * 70)

for rank, index_number in enumerate(indices[0], start=1):

    print(f"\nResult {rank}")
    print("-" * 70)

    print(f"Chunk index: {index_number}")
    print(f"Distance: {distances[0][rank - 1]}")

    print("\nChunk:")
    print(chunks[index_number])