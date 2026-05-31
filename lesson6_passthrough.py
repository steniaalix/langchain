from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

parser=StrOutputParser()

explanation_prompt=ChatPromptTemplate.from_template("Explain this topic simply: {topic}")

explanation_chain=explanation_prompt|llm|parser

chain=RunnableParallel({
    "original_topic":RunnablePassthrough(),
    "explanation":{"topic":RunnablePassthrough()}|explanation_chain
})

response=chain.invoke("Python Inheritance")

print(response)