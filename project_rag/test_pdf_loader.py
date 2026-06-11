from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR/"pdfs"

pdf_files = list(PDF_FOLDER.glob("*pdf"))

if not pdf_files:
    print("No PDF Files found inside the pdfs folder.")
    exit()

pdf_path = pdf_files[0]

print("PDF selected:")
print(pdf_path)

loader = PyPDFLoader(str(pdf_path))

documents = loader.load()

print(f"\nLoaded {len(documents)} PDF page documents.")

for index , document in enumerate(documents , start=1):
    print(f"\n--- Page Document {index} ---")
    print("Metadata:")
    print(document.metadata)

    print("\nContent preview:")
    print(document.page_content[:500])