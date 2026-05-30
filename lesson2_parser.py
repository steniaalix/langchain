from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

prompt=ChatPromptTemplate.from_messages([("system", "You are a helpful AI tutor. Explain clearly with examples."),("human", "Teach me about {topic}.")])

parser=StrOutputParser()
chain=  prompt|llm|parser
response=chain.invoke({"topic":"Python Dictionaries"})
print(response)