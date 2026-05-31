'''
from langchain_core.runnables import RunnableLambda

def add_exclamation(text):
    return text+"!"

def make_upper(text):
    return text.upper()
chain=RunnableLambda(add_exclamation)|RunnableLambda(make_upper)

response=chain.invoke("Hello")
print(response)
'''

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

def clean_input(topic):
    return {"topic":topic.strip().lower()}

cleaner=RunnableLambda(clean_input)

prompt=ChatPromptTemplate.from_template("Explain {topic} in a simple way")

chain=cleaner|prompt|llm|StrOutputParser()

response=chain.invoke("   PYTHON Functions  ")
print(response)