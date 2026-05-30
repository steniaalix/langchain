from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()
llm=ChatGroq(model="llama-3.3-70b-versatile")
sysmessage=input("Enter the character you want the ai to be:")
messages=[SystemMessage(content=f"you are {sysmessage}.Stay in this character for the whole onversation.")]
choice="yes"
while choice=="yes":
    hummessage=input("Enter your prompt:")


    messages.append(HumanMessage(content=hummessage))

    response=llm.invoke(messages)
    print(response.content)
    print()
    print()
    messages.append(AIMessage(content=response.content))
    choice=input("Do you want to coontinue?yes/no:").lower()