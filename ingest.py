from pypdf import PdfReader

# PDF file name
pdf_path = "HR_policy.pdf"

# Read the PDF
reader = PdfReader(pdf_path)

print(f"Number of pages: {len(reader.pages)}")

# Store all extracted text here
all_text = ""

# Extract text from every page
for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:
        all_text += text + "\n"

print("\nPDF ingestion completed.")

print(f"Total characters extracted: {len(all_text)}")

print("\nFirst 1000 characters of extracted text:")
print("-" * 50)
print(all_text[:1000])
print("-" * 50)