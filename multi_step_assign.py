from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.8)

parser=StrOutputParser()

explanation_prompt=ChatPromptTemplate.from_template(
    "Explain {topic} simply for a {level} student."
)

quiz_prompt=ChatPromptTemplate.from_template(
    """
Create one quiz question based on this explanation.

Topic: {topic}
Explanation: {explanation}
"""
)

study_plan_prompt=ChatPromptTemplate.from_template(
    """
Create a short study plan.

Topic: {topic}
Level: {level}
Explanation: {explanation}
Quiz: {quiz}
"""
)

explanation_chain=explanation_prompt|llm|parser

quiz_chain=quiz_prompt|llm|parser

study_plan_chain=study_plan_prompt|llm|parser

chain=(
    RunnablePassthrough.assign(
    explanation=explanation_chain
)
|RunnablePassthrough.assign(
    quiz=quiz_chain
)
|RunnablePassthrough.assign(
    study_plan=study_plan_chain
)
)
response=chain.invoke({
    "topic":"Python OOP",
    "level": "beginner"
})

print("\nTopic:")
print(response["topic"])

print("\nExplanation:")
print(response["explanation"])

print("\nQuiz:")
print(response["quiz"])

print("\nStudy Plan:")
print(response["study_plan"])