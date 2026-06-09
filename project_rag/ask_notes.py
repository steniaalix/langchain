from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()


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


retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
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


# -----------------------------
# 6. Helper function for formatting context
# -----------------------------

def format_context(retrieved_docs):
    context_parts = []

    for doc in retrieved_docs:
        source_path = doc.metadata.get("source", "unknown source")
        source_name = Path(source_path).name

        context_parts.append(
            f"Source: {source_name}\nContent:\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


# -----------------------------
# 7. Helper function for printing sources
# -----------------------------

def print_sources(retrieved_docs):
    unique_sources = []

    for doc in retrieved_docs:
        source_path = doc.metadata.get("source", "unknown source")
        source_name = Path(source_path).name

        if source_name not in unique_sources:
            unique_sources.append(source_name)

    print("\nSources used:")
    for index, source in enumerate(unique_sources, start=1):
        print(f"{index}. {source}")


# -----------------------------
# 8. Ask questions
# -----------------------------

while True:
    question = input("\nAsk a question from your notes or type exit: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Bye!")
        break

    retrieved_docs = retriever.invoke(question)

    context = format_context(retrieved_docs)

    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAI Answer:")
    print(answer)

    print_sources(retrieved_docs)