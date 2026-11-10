import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Load API key
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY was not found in .env")

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# 2. Read the PDF
# --------------------------------------------------

pdf_path = "HR_policy.pdf"

reader = PdfReader(pdf_path)

all_text = ""

for page in reader.pages:

    text = page.extract_text()

    if text:
        all_text += text + "\n"


# --------------------------------------------------
# 3. Split the text into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(all_text)

print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# 4. Create embeddings for ALL chunks
# --------------------------------------------------

embeddings = []

for i, chunk in enumerate(chunks):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )

    embedding = response.data[0].embedding

    embeddings.append(embedding)

    print(f"Embedded chunk {i + 1}/{len(chunks)}")


# --------------------------------------------------
# 5. Check the result
# --------------------------------------------------

print("\nEmbedding process completed.")

print(f"Number of chunks: {len(chunks)}")
print(f"Number of embeddings: {len(embeddings)}")
print(f"Dimensions of each embedding: {len(embeddings[0])}")

import pickle

# Save chunks and embeddings
data = {
    "chunks": chunks,
    "embeddings": embeddings
}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("\nSaved embeddings to embeddings.pkl")