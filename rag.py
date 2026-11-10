import os
import pickle
import numpy as np
import faiss

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in .env")

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# 2. Load chunks
# --------------------------------------------------

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]


# --------------------------------------------------
# 3. Load FAISS index
# --------------------------------------------------

index = faiss.read_index("faiss.index")


# --------------------------------------------------
# 4. Ask user a question
# --------------------------------------------------

question = input("\nEnter your question: ")


# --------------------------------------------------
# 5. Create embedding for the question
# --------------------------------------------------

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

question_embedding = response.data[0].embedding


# --------------------------------------------------
# 6. Convert question embedding to NumPy
# --------------------------------------------------

query_vector = np.array(
    [question_embedding],
    dtype="float32"
)


# --------------------------------------------------
# 7. Search FAISS
# --------------------------------------------------

k = 5

distances, indices = index.search(
    query_vector,
    k
)


# --------------------------------------------------
# 8. Collect retrieved chunks
# --------------------------------------------------

retrieved_chunks = []
for rank, index_number in enumerate(indices[0], start=1):

    print(f"\n--- Retrieved Chunk {rank} ---")
    print(f"Chunk index: {index_number}")
    print(f"Distance: {distances[0][rank - 1]}")
    print(chunks[index_number])

    retrieved_chunks.append(chunks[index_number])



# Combine retrieved chunks
context = "\n\n---\n\n".join(retrieved_chunks)


# --------------------------------------------------
# 9. Send question + retrieved context to GPT
# --------------------------------------------------

prompt = f"""
Answer the user's question using ONLY the context
provided below.

If the answer is not present in the context,
say that the information was not found in the
provided document.

Context:
{context}

User question:
{question}
"""

print("\nGenerating answer...")


response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)


# --------------------------------------------------
# 10. Display final answer
# --------------------------------------------------

print("\nAnswer:")
print("=" * 70)
print(response.output_text)