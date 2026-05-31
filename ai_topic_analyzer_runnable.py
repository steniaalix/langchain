from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

parser=StrOutputParser()


explanation_prompt=ChatPromptTemplate.from_template("Give simple explanation on {topic}")
analogy_prompt=ChatPromptTemplate.from_template("Give one real-world analogy  for {topic}")
mistake_prompt=ChatPromptTemplate.from_template("Give two common mistakes made on {topic}")
quiz_prompt=ChatPromptTemplate.from_template("Create a beginner-friendly quiz question for {topic}")


explanation_chain=explanation_prompt|llm|parser
analogy_chain=analogy_prompt|llm|parser
mistake_chain=mistake_prompt|llm|parser
quiz_chain=quiz_prompt|llm|parser

parallel_chain=RunnableParallel({
    "explanation":explanation_chain,
    "analogy":analogy_chain,
    "mistake":mistake_chain,
    "quiz":quiz_chain
})

response=parallel_chain.invoke({
    "topic":"Python OOP"})


print("\nExplanation:")
print(response["explanation"])

print("\nAnalogy:")
print(response["analogy"])

print("\n Common mistakes:")
print(response["mistake"])

print("\nQuiz:")
print(response["quiz"])