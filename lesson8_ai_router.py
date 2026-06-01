from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9)

parser=StrOutputParser()

explanation_prompt = ChatPromptTemplate.from_template(
    """
You are a clear AI tutor.

Explain this topic simply:

User request: {request}
"""
)

quiz_prompt = ChatPromptTemplate.from_template(
    """
You are a quiz generator.

Create 3 beginner-friendly quiz questions based on this request:

User request: {request}
"""
)

project_prompt = ChatPromptTemplate.from_template(
    """
You are a project mentor.

Suggest one practical beginner project based on this request:

User request: {request}
"""
)

default_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful study assistant.

The user's request is unclear, but try to help them:

User request: {request}
"""
)
explanation_chain=explanation_prompt|llm|parser
quiz_chain=quiz_prompt|llm|parser
project_chain=project_prompt|llm|parser
default_chain=default_prompt|llm|parser

def is_quiz(input):
    return "quiz" in input["request"].lower()

def is_project(input):
    return "project" in input["request"].lower()

def is_explanation(input):
    text=input["request"].lower()
    return "explain" in text or "teach" in text or "what" in text

router_chain=RunnableBranch(
    (is_quiz,quiz_chain),
    (is_project,project_chain),
    (is_explanation,explanation_chain),
    default_chain
)

user_request=input("What do you want? ").strip()

response=router_chain.invoke({
    "request":user_request
})

print("\nAI Response:\n")
print(response)