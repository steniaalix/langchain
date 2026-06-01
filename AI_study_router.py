from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9)

parser=StrOutputParser()

explanation_prompt = ChatPromptTemplate.from_template(
    """
You are an expert AI tutor.

Explain clearly and simply.

User request: {request}
"""
)

quiz_prompt = ChatPromptTemplate.from_template(
    """
You are an expert quiz creator.

Create 5 useful quiz questions.

User request: {request}
"""
)

project_prompt = ChatPromptTemplate.from_template(
    """
You are an expert project mentor.

Suggest one practical beginner project.
Include:
- project idea
- required skills
- simple steps

User request: {request}
"""
)

study_plan_prompt = ChatPromptTemplate.from_template(
    """
You are an expert study planner.

Create a practical 5-step study plan.

User request: {request}
"""
)

default_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI study assistant.

The request does not clearly match explanation, quiz, project, or study plan.
Still give a useful helpful response.

User request: {request}
"""
)

explanation_chain=explanation_prompt|llm|parser

quiz_chain=quiz_prompt|llm|parser

project_chain=project_prompt|llm|parser

study_plan_chain=study_plan_prompt|llm|parser

default_chain=default_prompt|llm|parser

def is_quiz(input):
    text=input["request"].lower()
    return (
    "quiz" in text
    or "test me" in text
    or "questions" in text
    or "question" in text
    )

def is_project(input):
    text=input["request"].lower()
    return (
        "project" in text 
        or "build" in text
        or "make an app" in text
        or "practice project" in text

    )

def is_study_plan_request(input_data):
    text = input_data["request"].lower()
    return (
        "study plan" in text
        or "roadmap" in text
        or "learn path" in text
        or "how should i study" in text
    )


def is_explanation_request(input_data):
    text = input_data["request"].lower()
    return (
        "explain" in text
        or "teach" in text
        or "what is" in text
        or "how does" in text
        or "understand" in text
    )


router_chain=RunnableBranch(
    (is_quiz,quiz_chain),
    (is_project,project_chain),
    (is_study_plan_request, study_plan_chain),
    (is_explanation_request, explanation_chain),
    default_chain
)

while True:
    user_request=input("\nsk something, or type exit: ").strip()

    if user_request.lower()=="exit":
        print("Bye!!")
        break

    response=router_chain.invoke({
        "request":user_request
    })

    print("\nAI Response:\n")
    print(response)
    print("\n"+"-"*60)