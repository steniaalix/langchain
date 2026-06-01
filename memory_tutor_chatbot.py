from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9)

prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI tutor. You have to remember the conversation and answer clearly"
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{input}"
    )
])


chain=prompt|llm

store={}

def get_session_history(session_id: str):
    if session_id.strip() not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory=RunnableWithMessageHistory(
    chain,
    get_session_history,
    user_messages_key="input",
    history_messages_key="history"
)
session_id=input("Enter your session id:").strip()
config={
    "configurable":{
        "session_id":session_id
    }
}

while True:
    user_input=input("\nYou:").strip()

    if user_input.lower() in ["exit","quit"]:
        print("Bye!!!")
        break

    if "history" in user_input.lower():

        history=get_session_history(session_id)
        print("\nConversation History:")
        for message in history.messages:
            print(type(message),__name__,':',message.content)

        continue


    response=chain_with_memory.invoke({
        "input":user_input
    },
    config=config
    )

    print("\nAI:")
    print(response.content)