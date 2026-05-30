from dotenv import load_dotenv
from langchain_core.prompts  import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

prompt=ChatPromptTemplate.from_messages([
    (
        "system",
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
    )
])
parser=StrOutputParser()

chain=prompt|llm|parser

inputs = [
    {
        "topic": "Python lists",
        "level": "beginner",
        "style": "simple"
    },
    {
        "topic": "Python dictionaries",
        "level": "beginner",
        "style": "friendly"
    },
    {
        "topic": "APIs",
        "level": "intermediate",
        "style": "technical"
    }
]

response=chain.batch(inputs)
for index,response in enumerate(response,start=1):
    print(f"\nResponse {index}")
    print("-"*50)
    print(response)