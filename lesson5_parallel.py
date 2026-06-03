from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

parser=StrOutputParser()

summary_prompt=ChatPromptTemplate.from_template("Explain {topic} in three simple sentences.")

analogy_prompt=ChatPromptTemplate.from_template("Give one analogy for {topic}")

quiz_prompt=ChatPromptTemplate.from_template("Create one beginnerr-friendly quiz question on {topic}")

summary_chain=summary_prompt|llm|parser

analogy_chain=analogy_prompt|llm|parser

quiz_chain=quiz_prompt|llm|parser

parallel_chain=RunnableParallel({
    "summary":summary_chain,
    "analogy":analogy_chain,
    "quiz":quiz_chain
})
response=parallel_chain.invoke({
    "topic":"Python functions"
})
print(response)


print("\nSummary:")
print(response["summary"])

print("\nAnalogy:")
print(response["analogy"])

print("\nQuiz:")
print(response["quiz"])