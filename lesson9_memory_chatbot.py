from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI tutor. Remember the conversation and answer clearly."
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{input}"
    )
])

chain = prompt | llm

store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

session_id = input("Enter session id: ").strip()

config = {
    "configurable": {
        "session_id": session_id
    }
}

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Bye!")
        break

    response = chain_with_history.invoke(
        {
            "input": user_input
        },
        config=config
    )

    print("\nAI:")
    print(response.content)

    history = get_session_history(session_id)
    print(history)
'''
    print("\n--- DEBUG MEMORY ---")
    for message in history.messages:
        print(type(message).__name__, ":", message.content)
    print("--------------------")
'''