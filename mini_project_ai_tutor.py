from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

parse=StrOutputParser()

prompt=ChatPromptTemplate.from_messages([(
        "system",
        """
You are an expert programming tutor.

Rules:
- Explain slowly.
- Use beginner-friendly language.
- Give one simple code example.
- Give one practice task.
- Do not make the answer too long.
"""),(
        "human",
        "Teach me this topic: {topic}"
    )])

chain=prompt|llm|parse

while True:
    topic = input("\nEnter a topic you want to learn, or type 'exit': ")

    if topic.lower().strip() == "exit":
        print("Goodbye!")
        break

    response = chain.invoke({
        "topic": topic
    })

    print("\nAI Tutor:\n")
    print(response)