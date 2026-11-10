from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Load the PDF
# --------------------------------------------------

pdf_path = "HR_policy.pdf"

reader = PdfReader(pdf_path)

all_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        all_text += text + "\n"


# --------------------------------------------------
# 2. Create the text splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# --------------------------------------------------
# 3. Split the document
# --------------------------------------------------

chunks = text_splitter.split_text(all_text)


# --------------------------------------------------
# 4. Display the results
# --------------------------------------------------

print(f"Total characters: {len(all_text)}")
print(f"Number of chunks: {len(chunks)}")

print("\nFirst 5 chunks:")
print("=" * 60)

for i, chunk in enumerate(chunks[:5], start=1):

    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print(f"\nCharacters in this chunk: {len(chunk)}")