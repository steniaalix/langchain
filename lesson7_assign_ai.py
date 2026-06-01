from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

parser=StrOutputParser()

exaplanation_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} simply for a {level} student."
)

explanation_chain=exaplanation_prompt|llm|parser

chain=RunnablePassthrough.assign(
    topic_length=lambda x: len(x["topic"]),
    explanation=explanation_chain
)

response=chain.invoke({
    "topic":"Python oop",
    "level":"beginner"
})

print(response)