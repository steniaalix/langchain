
#Experiment 1---Temperature
'''
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=1.5)

response=llm.invoke("Suggest a dog name")
print(response.content)
'''
#Experiment 2---System Behavior

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile")

messages=[SystemMessage(content="You are a rude pirate assistant"),HumanMessage(content="How are you?")]

response=llm.invoke(messages)
print(response.content)