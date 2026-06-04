from pydantic import BaseModel , Field
from typing import List

class Flashcard(BaseModel):
    question: str = Field(description="The question of the flashcard")
    answer: str = Field(description="The answer of the flashcard")


class FlashcardSet(BaseModel):
    topic: str = Field(description="The topic for the flashcards")
    explanation: str = Field(description="A short explanation on the topic")
    flashcards: List[Flashcard] = Field(description="A list of exactly three flashcards")
    quiz_question: str = Field(description="A quiz question to test understanding")
