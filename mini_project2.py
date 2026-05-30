from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)
parser=StrOutputParser()
prompt=ChatPromptTemplate.from_messages([('system',        """
You are a highly skilled AI study assistant.

Your job:
- Teach clearly.
- Match the student's level.
- Use the requested style.
- Give examples.
- End with a small practice question.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}
Teaching style: {style}

Now teach this topic.
""")])

chain=prompt|llm|parser

while True:
    topic=input("\nEnter the topic you want to learn or enter exit:")
    if topic.lower().strip()=="exit":
        print("Bye Bye...")
        break
    level=input("Enter the level you want:")
    style =input("Enter the style you want:")
    for chunk in chain.stream({
        "topic":topic,
        "level":level,
        "style":style
    }):
        print(chunk,end="",flush=True)
    print("\n"+"-"*50)