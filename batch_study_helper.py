from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a highly skilled AI study assistant.

Rules:
- Explain clearly.
- Match the student's level.
- Use the requested style.
- Give one example.
- End with one practice question.
- Keep the answer short.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}
Teaching style: {style}

Teach this topic.
"""
    )
])

chain=prompt|llm|StrOutputParser()

topics_text=input("Enter topics seerated by commaa:")
level=input("Enter the level:")
style=input("Enter the style:")

topics= topics_text.split(",")

inputs=[]
for topic in topics:
    inputs.append({"topic":topic.strip(),"level":level.strip(),"style":style.strip()})


'''
responses=chain.batch(inputs,config={"max_concurrency":2})

for topic,response in zip(topics,responses):
    print("\n"+"="*60)
    print(f"Topic : {topic.strip()}")
    print("="*60)
    print(response)

'''
responses=chain.batch(inputs,return_exceptions=True)

for index,response in enumerate(responses,start=1):
    print(f"\nResult {index}")
    print("-"*50)

    if isinstance(response, Exception):
        print("Error:")
        print(response)

    else:
        print(response)