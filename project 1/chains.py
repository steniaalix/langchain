from config import (
    NORMAL_MODEL,
    STRUCTURED_MODEL,
    NORMAL_TEMPERATURE,
    STRUCTURED_TEMPERATURE
)
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from schemas import FlashcardSet
from memory_manager import get_session_history


load_dotenv()


llm = ChatGroq(
    model=NORMAL_MODEL,
    temperature=NORMAL_TEMPERATURE
)

structure_based_llm = ChatGroq(
    model=STRUCTURED_MODEL,
    temperature=STRUCTURED_TEMPERATURE
)

parser = StrOutputParser()


# -----------------------------
# Explanation Chain
# -----------------------------

exp_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI tutor.

Rules:
- Explain according to the student's level.
- Give a simple and short explanation.
- Be precise.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}

Explain this topic clearly.
"""
    )
])

exp_chain = exp_prompt | llm | parser


# -----------------------------
# Flashcard Chain
# -----------------------------

structured_llm = structure_based_llm.with_structured_output(FlashcardSet)

flashcard_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI teacher who prepares flashcards.

Rules:
- Create exactly three flashcards.
- Give a short explanation on the topic before flashcards.
- Be clear and simple.
- Match the student's level.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}

Generate the flashcard set.
"""
    )
])

flashcard_chain = flashcard_prompt | structured_llm


# -----------------------------
# Batch Flashcards Chain
# -----------------------------

batch_flashcards_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI teacher who prepares flashcards for all the topics given.

Rules:
- Create exactly three flashcards for each topic.
- Give a short explanation on the topic before flashcards.
- Be clear and simple.
- Match the student's level.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Level: {level}
"""
    )
])

batch_flashcards_chain = batch_flashcards_prompt | structured_llm


# -----------------------------
# Quiz Chain
# -----------------------------

quiz_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI tutor who creates quizzes on the given topic according to the student's level.

Rules:
- Make exactly three quizzes.
- Be clear and simple.
- Give a short explanation before giving the quiz.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Level: {level}
"""
    )
])

quiz_chain = quiz_prompt | llm | parser


# -----------------------------
# Project Chain
# -----------------------------

project_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI assistant who gives project ideas based on the topic and level.

Rules:
- Be clear and simple.
- Give exactly three project ideas.
- Match the user's level.
- Give a short explanation of each project.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Level: {level}
"""
    )
])

project_chain = project_prompt | llm | parser


# -----------------------------
# Analyzer Chains
# -----------------------------

analogy_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI assistant who gives real-life analogies based on the topic.

Rules:
- Give a real-life analogy.
- Match the user's level.
- Be clear and simple.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Level: {level}
"""
    )
])

analogy_chain = analogy_prompt | llm | parser


mistake_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI assistant who gives a common mistake for a topic.

Rules:
- Give a common mistake based on the topic.
- Match the user's level.
- Be clear and simple.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Level: {level}
"""
    )
])

mistake_chain = mistake_prompt | llm | parser


analyze_chain = RunnableParallel({
    "explanation": exp_chain,
    "analogy": analogy_chain,
    "mistake": mistake_chain,
    "project": project_chain,
    "quiz": quiz_chain
})


# -----------------------------
# Study Pack Chain using .assign()
# -----------------------------

study_plan_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI study assistant. Create a study plan based on all given information.

Rules:
- Be clear and simple.
- Match the user's level.
- Use all given information.
- Be friendly.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Level: {level}

Explanation:
{explanation}

Analogy:
{analogy}

Common Mistake:
{mistake}

Quiz:
{quiz}

Project Ideas:
{project}

Create a practical study plan.
"""
    )
])

study_plan_chain = study_plan_prompt | llm | parser

study_pack_chain = (
    RunnablePassthrough.assign(
        explanation=exp_chain
    )
    | RunnablePassthrough.assign(
        analogy=analogy_chain
    )
    | RunnablePassthrough.assign(
        mistake=mistake_chain
    )
    | RunnablePassthrough.assign(
        quiz=quiz_chain
    )
    | RunnablePassthrough.assign(
        project=project_chain
    )
    | RunnablePassthrough.assign(
        study_plan=study_plan_chain
    )
)


# -----------------------------
# Chat Memory Chain
# -----------------------------

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI study companion who remembers the conversation.

Rules:
- Remember the conversation.
- Answer follow-up questions using chat history.
- Help the student learn clearly.
- If the student asks what they said earlier, use the history.
"""
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{input}"
    )
])

chat_chain = chat_prompt | llm

memory_chatbot = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


# -----------------------------
# Default Chain
# -----------------------------

default_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI study assistant."
    ),
    (
        "human",
        """
The command is unclear.

User command: {command}

Tell the user to use:
- explain
- flashcards
- batch flashcards
- quiz
- project
- history
- analyze
- study pack
- chat
- chat history
- clear memory
- exit
"""
    )
])

default_chain = default_prompt | llm | parser


# -----------------------------
# Router Conditions
# -----------------------------

def is_exp(input_data):
    return input_data["command"] == "explain"


def is_flashcard(input_data):
    return input_data["command"] == "flashcards"


def is_quiz(input_data):
    return input_data["command"] == "quiz"


def is_project(input_data):
    return input_data["command"] == "project"


def is_analyze(input_data):
    return input_data["command"] == "analyze"


def is_study_pack(input_data):
    return input_data["command"] == "study pack"


branch_chain = RunnableBranch(
    (is_exp, exp_chain),
    (is_flashcard, flashcard_chain),
    (is_quiz, quiz_chain),
    (is_project, project_chain),
    (is_analyze, analyze_chain),
    (is_study_pack, study_pack_chain),
    default_chain
)