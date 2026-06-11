import os
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader


BASE_DIR=Path(__file__).resolve().parent

PDF_FOLDER = BASE_DIR/"pdfs"
NOTES_FOLDER = BASE_DIR/"notes"
CHROMA_FOLDER = BASE_DIR/"chroma_db"


# -----------------------------
# 1. Load all .txt files
# -----------------------------

documents = []

notes_path = NOTES_FOLDER
pdfs_path = PDF_FOLDER

if not notes_path.exists():
    print(f"Folder '{NOTES_FOLDER}' does not exist.")
    print("Please create a notes folder and add .txt files inside it.")
    exit()

if not pdfs_path.exists():
    print(f"Folder '{PDF_FOLDER}' does not exist.")
    print("Please create a pdfs folder and add .pdf files inside it. ")
    exit()

    
txt_files = list(notes_path.glob("*.txt"))

pdf_files= list(PDF_FOLDER.glob("*.pdf"))

if not txt_files:
    print(f"No .txt files found inside '{NOTES_FOLDER}'.")
    exit()

if not pdf_files:
    print(f"\n No pdf files found inside  '{PDF_FOLDER}'.")
    exit()

for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    document = Document(
        page_content=text,
        metadata={
            "source": str(file_path)
        }
    )

    documents.append(document)

print(f"Loaded {len(txt_files)} Text Files.")


for pdf_path in pdf_files:
    loader= PyPDFLoader(str(pdf_path))
    pdf_documents = loader.load()

    documents.extend(pdf_documents)

print(f"Loaded {len(pdf_files)} PDF Files.")

if not documents :
    print("No documents found.")
    print("Please add .txt files inside notes/ or .pdf files inside pdfs/.")
    exit()

# -----------------------------
# 2. Split documents
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)


print(f"Created {len(chunks)} chunks.")


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Clear old vector database
# -----------------------------

if CHROMA_FOLDER.exists():
    shutil.rmtree(CHROMA_FOLDER)
    print("Old chroma_db deleted.")

vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_FOLDER)
)


# -----------------------------
# 5. Store chunks in Chroma
# -----------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_FOLDER
)

print("Vector database created successfully.")
print(f"Saved inside folder: {CHROMA_FOLDER}")