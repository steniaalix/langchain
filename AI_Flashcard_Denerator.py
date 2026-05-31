from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

prompt=ChatPromptTemplate.from_messages([
    (
        "system",
    """
You are an expert study assistant specializing in active recall and spaced repetition.

Create flashcards that help students remember and understand concepts deeply.

Rules:
- Adapt to the student's level.
- Prefer conceptual questions over shallow definition-only questions.
- Use clear question-answer flashcard format.
- Keep each answer concise.
- Include one practical or application-based question when possible.
- Do not include unnecessary introductions.
- Do not say "Sure" or "Here you go".
"""
), (
        "human",
        """
Topic: {topic}
Student level: {level}

Create:
1. A short explanation
2. Three flashcards
3. One quiz question
"""
    )
])

chain = prompt|llm|StrOutputParser()

topics_text=input("Enter the topics seperated by commas:")
level=input("Enter your level:")

topics=topics_text.split(",")

inputs=[]
for topic in topics:
    inputs.append({
        "topic":topic.strip(),
        "level":level.strip()
    })
responses=chain.batch(inputs)

for topic , response in zip(topics,responses):
    print("="*60)
    print(f"\n Topic: {topic.strip()}")
    print("="*60)

    print(response)