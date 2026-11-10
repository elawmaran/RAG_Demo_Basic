import pickle
import numpy as np
import faiss


# 1. Load saved chunks and embeddings
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
embeddings = data["embeddings"]

print(f"Number of chunks: {len(chunks)}")
print(f"Number of embeddings: {len(embeddings)}")


# 2. Convert embeddings into a NumPy array
embedding_matrix = np.array(embeddings, dtype="float32")

print(f"Embedding matrix shape: {embedding_matrix.shape}")


# 3. Get embedding dimensions
dimension = embedding_matrix.shape[1]

print(f"Embedding dimensions: {dimension}")


# 4. Create FAISS index
index = faiss.IndexFlatL2(dimension)

print("\nFAISS index created.")


# 5. Add embeddings to FAISS
index.add(embedding_matrix)

print(f"Number of vectors in FAISS: {index.ntotal}")


# 6. Save FAISS index
faiss.write_index(index, "faiss.index")

print("\nFAISS index saved successfully.")