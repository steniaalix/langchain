from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough,RunnableBranch,RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel , Field
from typing import List


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.9
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
# Flashcard Structured Output
# -----------------------------

class Flashcard(BaseModel):
    question: str = Field(description="The question of the flashcard")
    answer: str = Field(description="The answer of the flashcard")


class FlashcardSet(BaseModel):
    topic: str = Field(description="The topic for the flashcards")
    explanation: str = Field(description="A short explanation on the topic")
    flashcards: List[Flashcard] = Field(description="A list of exactly three flashcards")
    quiz_question: str = Field(description="A quiz question to test understanding")


structured_llm = llm.with_structured_output(FlashcardSet)

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


branch_chain = RunnableBranch(
    (is_exp, exp_chain),
    (is_flashcard, flashcard_chain),
    default_chain
)


# -----------------------------
# Helper Function to Print Flashcards
# -----------------------------

def print_flashcards(response):
    print("\n" + "=" * 60)
    print(f"Topic: {response.topic}")
    print("=" * 60)

    print("\nExplanation:")
    print(response.explanation)

    print("\nFlashcards:")
    for index, card in enumerate(response.flashcards, start=1):
        print(f"\nFlashcard {index}")
        print(f"Q: {card.question}")
        print(f"A: {card.answer}")

    print("\nQuiz Question:")
    print(response.quiz_question)


# -----------------------------
# Main App
# -----------------------------

while True:
    command = input("\nEnter command explain/flashcards/exit: ").strip().lower()

    if command == "exit":
        print("Bye!")
        break

    topic = input("Enter the topic: ").strip()
    level = input("Enter the level: ").strip()

    input_data = {
        "command": command,
        "topic": topic,
        "level": level
    }

    response = branch_chain.invoke(input_data)

    if isinstance(response, FlashcardSet):
        print_flashcards(response)
    else:
        print("\nAI:")
        print(response)

    print("\n" + "-" * 60)