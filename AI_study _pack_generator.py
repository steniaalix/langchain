from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)
parser=StrOutputParser()
exp_prompt=ChatPromptTemplate.from_template(
    "Give simple explanatino on {topic} for a {level} student"
)

exp_chain=exp_prompt|llm|parser

analogy_prompt=ChatPromptTemplate.from_template(
    """Give one real-life using these topic and explantion
    
    Topic: {topic}
    Explanation: {explanation}"""
)

analogy_chain=analogy_prompt|llm|parser


mistake_prompt=ChatPromptTemplate.from_template(
    """Give a common mistake using the topic , explanation and analogy
    
    Topic: {topic}
    Explanation: {explanation}
    Analogy: {analogy}
    """
)


mistake_chain=mistake_prompt|llm|parser

quiz_prompt=ChatPromptTemplate.from_template(
    """
Create a simple quiz based on the topic,explanation,analogy,common mistake:

Topic: {topic}
Explanation: {explanation}
Analogy: {analogy}
Common Mistake: {mistake}
"""
)

quiz_chain=quiz_prompt|llm|parser


study_plan_prompt=ChatPromptTemplate.from_template(
    """
Create a detailed study plan based on the topic,explanatin,analogy,common mistake,quiz:

Topic: {topic}
Explanation: {explanation}
Common Mistake: {mistake}
Quiz: {quiz}"""
)

study_plan_chain=study_plan_prompt|llm|parser

chain=(
    RunnablePassthrough.assign(
        explanation=exp_chain
    )
    |RunnablePassthrough.assign(
        analogy=analogy_chain
    )
    |RunnablePassthrough.assign(
        mistake=mistake_chain
    )
    |RunnablePassthrough.assign(
        quiz=quiz_chain
    )
    |RunnablePassthrough.assign(
        study_plan=study_plan_chain
    )
)

topic=input("Enter the topic :").strip()

level=input("Enter the level :")

response=chain.invoke({
    "topic":topic,
    "level":level
})

print('='*60)
print("\nTopic:")
print(response["topic"])
print('='*60)
print("")
print("\nExplanation:")
print(response["explanation"])
print("")
print("")
print("\nAnalogy:")
print(response["analogy"])
print("")
print("")
print("\nCommon Mistake:")
print(response["mistake"])
print("")
print("")
print("\nQuiz:")
print(response["quiz"])
print("")
print("")
print("\nStudy Plan:")
print(response["study_plan"])