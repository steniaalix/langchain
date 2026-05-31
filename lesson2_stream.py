from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

prompt=ChatPromptTemplate.from_messages([("system",
        """
You are a highly skilled AI study assistant.

Your job:
-Teach clearly
-Match the student's level.
-Use the requested sstyle.
Give examples.
-End with a small practice question."""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}
Teaching style: {style}

Now teach this topic.
        """
    )])

chain = prompt|llm|StrOutputParser()


for chunk in chain.stream({
    "topic": "Python lists",
    "level": "beginner",
    "style": "simple"
}):
    print(chunk,end="",flush=True)