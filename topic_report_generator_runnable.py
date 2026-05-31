from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

exp_prompt=ChatPromptTemplate.from_template("Give a simple explanation on {topic}")
analogy_prompt=ChatPromptTemplate.from_template("Give a real-world analogy of {topic}")
task_prompt=ChatPromptTemplate.from_template("Give a simple practice task on {topic}")

parser=StrOutputParser()

exp_chain=exp_prompt|llm|parser
analogy_chain=analogy_prompt|llm|parser
task_chain=task_prompt|llm|parser

chain=RunnableParallel({
    "original":RunnablePassthrough(),
    "explanation":{"topic":RunnablePassthrough()}|exp_chain,
    "analogy":{"topic":RunnablePassthrough()}|analogy_chain,
    "task":{"topic":RunnablePassthrough()}|task_chain
})

topic=input("Enter the topic : ").strip()

response=chain.invoke(topic)
print("\nOriginal:")
print(response["original"])

print("\nSimple Explanation:")
print(response["explanation"])

print("\nAnalogy:")
print(response["analogy"])


print("\nPractice task:")
print(response["task"])