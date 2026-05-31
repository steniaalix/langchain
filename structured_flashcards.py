from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class Flashcard(BaseModel):
    question: str= Field(description="The flashcard question")
    answer: str=Field(description="The flashcard answer")

class FlashcardSet(BaseModel):
    topic:str=Field(description="The topic of the flashcards")
    explanation:str=Field(description="A short explanation of the topic")
    flashcards: List[Flashcard]=Field(description="A list of flashcards")
    quiz_question: str=Field(description="One quiz question to test understanding")

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)
structured_llm=llm.with_structured_output(FlashcardSet)

prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert study assistant specializing in active recall and spaced repetition.

Create useful flashcards that help students understand concepts deeply.

Rules:
- Match the student's level.
- Keep the explanation short.
- Create exactly three flashcards.
- Each flashcard must have a question and answer.
- Create one quiz question.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}

Create a structured flashcard set.
"""
    )
])

chain = prompt|structured_llm
response=chain.invoke({
    "topic":"Python lists",
    "level": "beginner"
})

print(response)
print("")
print("")
print("")
print("")
data=response.model_dump()

json_data=response.model_dump_json(indent=2)

print(data)
print("")
print("")
print("")
print("")
print(json_data)
print("")
print("")
print("")
print("")
print(f"\nTopic: {response.topic}")
print(f"\nExplanation: \n{response.explanation}")

print("\nFlashcards:")
for index,card in enumerate(response.flashcards,start=1):
    print(f"\nCard {index}")
    print("Q:", card.question)
    print("A:",card.answer)

print(f"\nQuiz: {response.quiz_question}")