from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

DEBUG=True
TOP_K=2
RELEVANCE_THRESHOLD=1.2

# -----------------------------
# 1. Set paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
CHROMA_FOLDER = BASE_DIR / "chroma_db"


# -----------------------------
# 2. Load embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 3. Load existing Chroma database
# -----------------------------

vector_store = Chroma(
    persist_directory=str(CHROMA_FOLDER),
    embedding_function=embeddings
)




# -----------------------------
# 4. Create LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)


# -----------------------------
# 5. Create RAG prompt
# -----------------------------

rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI tutor.

Use only the given context to answer the question.
If the answer is not in the context, say:
"I could not find that in the notes."

Do not use outside knowledge.
Be clear and simple.
"""
    ),
    (
        "human",
        """
Context:
{context}

Question:
{question}

Answer:
"""
    )
])


rag_chain = rag_prompt | llm | StrOutputParser()

#----------------------------------------
#Helper to get the page_label
#----------------------------------------

def get_source_label(doc):
    source_path=doc.metadata.get("source","unknown source")
    source_name=Path(source_path).name

    page_label= doc.metadata.get("page_label")
    if page_label :
        return f"{source_name}, page {page_label}"
    
    return source_name

# -----------------------------
# 6. Helper function for formatting context
# -----------------------------

def format_context(retrieved_docs):
    context_parts = []

    for doc in retrieved_docs:
        source_label=get_source_label(doc)
        context_parts.append(
            f"Source: {source_label}\nContent:\n{doc.page_content} "
        )
    return "\n\n---\n\n".join(context_parts)


# -----------------------------
# 7. Helper function for printing sources
# -----------------------------

def print_sources(retrieved_docs):
    unique_sources = []

    for doc in retrieved_docs:
        source_label= get_source_label(doc)

        if source_label not in unique_sources:
            unique_sources.append(source_label)

    print("\nSources used:")
    for index, source in enumerate(unique_sources, start=1):
        print(f"{index}. {source}")

#--------------------------------------
#Helper to print the chunks retrieved 
#--------------------------------------

def print_retrieved_docs(retrieved_docs):
    print("\nRetrieved chunks:")

    for index , doc in enumerate(retrieved_docs,start=1):
        source_path=doc.metadata.get("source","unknown source")
        source_name=Path(source_path).name

        print(f"\n--- Chunk {index} ---")
        print(f"Source: {source_name}")
        print("Content:")
        print(doc.page_content)


#--------------------------------------
#Helper to delete the duplicate chunks
#--------------------------------------

def remove_duplicate_docs(retrieved_docs):
    unique_docs=[]
    seen_content=set()

    for doc in retrieved_docs:
        content=doc.page_content.strip()

        if content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(content)
    
    return unique_docs

#---------------------------------
#Helper to filter relevant chunks
#---------------------------------

def filter_relevant_results(results_with_scores):
    relevant_docs=[]

    for doc, score in results_with_scores:
        if score <= RELEVANCE_THRESHOLD:
            relevant_docs.append(doc)

    return relevant_docs
 
 #------------------------------
 #Helper to print docs with scores
 #------------------------------

def print_results_with_scores(results_with_scores):
    print("\nRetrieved chunks with scores:")
    for index , (doc, score) in enumerate(results_with_scores,start=1):
        source_path= doc.metadata.get("source","unknown source")
        source_name= Path(source_path).name

        print(f"\n--- Chunk {index} ---")
        print(f"Source: {source_name}")
        print(f"Score: {score}")
        print("Content:")
        print(doc.page_content)
 
# -----------------------------
# 8. Ask questions
# -----------------------------

while True:
    question = input("\nAsk a question from your notes or type exit: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Bye!")
        break

    if not question:
        print("Question cannot be empty.")
        continue

    results_with_scores=vector_store.similarity_search_with_score(
        question,
        k=TOP_K
    )
    retrieved_docs= filter_relevant_results(results_with_scores)
    retrieved_docs=remove_duplicate_docs(retrieved_docs)

    if DEBUG:
        print_results_with_scores(results_with_scores)

    if not retrieved_docs:
        print("\nAI Answer:")
        print("I could not find that in the notes.")
        continue
    
    if DEBUG:
        print_retrieved_docs(retrieved_docs)



    context = format_context(retrieved_docs)
    print("\nAI Answer:")

    for chunk in rag_chain.stream({
        "context": context,
        "question": question}):
        print(chunk,end="",flush=True)
    

    print_sources(retrieved_docs)