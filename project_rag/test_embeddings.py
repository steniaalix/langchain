from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "What is inheritance?"

vector = embeddings.embed_query(text)

print("Text:", text)
print("Vector type:", type(vector))
print("Vector length:", len(vector))
print("First 10 numbers:")
print(vector[:10])