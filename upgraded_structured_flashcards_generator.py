from dotenv import load_dotenv
from pydantic import BaseModel , Field
from typing import List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class Flashcard(BaseModel):
    question:str=Field(description="The flashcard question")
    answer:str=Field(description="The flashcard answer")

class FlashcardSet(BaseModel):
    topic:str=Field(description="The topic of the flashcards")
    explanation:str=Field(description="A short explanatino of the topic")
    flashcards:List[Flashcard]=Field(description="A list of exactly three flashcards")
    quiz_question:str=Field(description="One quiz to test the understanding")

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

structured_llm=llm.with_structured_output(FlashcardSet)

prompt=ChatPromptTemplate.from_messages([
    ("system",
     """
You are an expert study assistant specializing in active recall and spaced repetion.


Rules:
-Adapt to the student's level
-Keep the explanation short
-Create exactly three flashcards
-Each flashcard must have a clear question and concise answer
-Create one quiz question that tests underrstanding 
-Do not include unnecessary introductions
"""
),(
    "human",
    """
Topic : {topic}
Studet level: {level}

Generate the flashcard set.
"""
)
])



chain=prompt|structured_llm

topic=input("Enter the topic:").strip()
level=input("Enter the level you want:").strip()

response=chain.invoke({
    "topic":topic,
    "level":level
})

print(f"\nTopic: {response.topic}")
print()
print(f"\nExplanation:")
print(response.explanation)

for index, card in enumerate(response.flashcards,start=1):
    print(f"\nFlashcard {index}")
    print(f"\nQ: {card.question}")
    print(f"\nA: {card.answer}")

print(f"\nQuiz:")
print(f"{response.quiz_question}")


