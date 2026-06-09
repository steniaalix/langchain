from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

with open("rag_notes.txt") as f:
    text=f.read()

document=Document(
    page_content=text,
    metadata={
        "source":"rag_notes.txt"
    }
)


documents=[document]

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks=splitter.split_documents(documents)

print(f"Loaded {len(documents)} document.")
print(f"Created {len(chunks)} chunks.")

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"    
)


vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)


retriever=vector_store.as_retriever(
    search_kwargs={
        "k":3
    }
)


llm=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
             )
rag_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI tutor.

Use only the given context to answer the question.

If the answer is not in the context, say:
"I could not find that in the notes.

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


rag_chain=rag_prompt|llm|StrOutputParser()



while True:
    question=input("\nAsk a question fron your notes or type exit: ").strip()
    if question.lower() in ["exit","quit"]:
        print("Bye!!!")
        break

    retrieved_docs=retriever.invoke(question)

    context="\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    answer=rag_chain.invoke({
        "context":context,
        "question":question
    })

    print("\nAI Answer:")
    print(answer)

    print("\nSources used:")
    for index, doc in enumerate(retrieved_docs,start=1):
        print(f"{index}, {doc.metadata}")