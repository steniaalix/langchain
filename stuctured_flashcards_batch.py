from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class Flashcard(BaseModel):
    question:str=Field(description="The question in flashcard")
    answer:str=Field(description="The answer in the flashcard")

class FlashcardSet(BaseModel):
    topic:str=Field(description="The topic for flashcard")
    explanation:str=Field(description="A short explanation on the topic")
    flashcards:List[Flashcard]=Field(description="List of exactly four flashcards")
    quiz:str=Field(description="A single quiz to test the understanding")


llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9)

structured_llm=llm.with_structured_output(FlashcardSet)

prompt=ChatPromptTemplate.from_messages([(
    "system",
    """
You are an expert AI tutor who teaches students clearly
Rules:
-Adapt to the student's level
-Be clear
-Give an short explanation on the topic
-Generate exactly four flashcards with question and answers
-Have a neat and short answers in the flashcards
-Give only one quiz on the topic to test the understanding"""
),(
    "human",
    """
Explain :
Topic: {topic}
Level:{level}"""
)])

chain=prompt|structured_llm

topic_text=input("Enter the topics you want to learn seperated by commas:").strip().split(",")
level=input("Enter the level:").strip()

inputs=[]
for topic in topic_text:
    inputs.append({
        "topic":topic,
        "level":level
    })

responses=chain.batch(inputs)

for response in responses:
    print("")
    print("")
    print("="*60)
    print("")
    print(f"\nTopic: {response.topic}")
    print("")
    print("="*60)
    print("")
    print("")
    print(f"\nExplanation: {response.explanation}")
    print("")
    
    for index,card in enumerate(response.flashcards,start=1):
        print(f"\nFlashcard {index}")
        print("")
        print("")
        print(f"Q: {card.question}")
        print("")
        print(f"\nA: {card.answer}")

print(f"\nQuiz:")
print(f"{response.quiz}")

