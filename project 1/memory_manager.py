import json
import os

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from config import MEMORY_FILE_NAME

chat_store = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, MEMORY_FILE_NAME)


def get_session_history(session_id: str):
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()

    return chat_store[session_id]


def message_to_dict(message):
    return {
        "type": message.type,
        "content": message.content
    }


def dict_to_message(data):
    if not isinstance(data, dict):
        return None

    message_type = data.get("type")
    content = data.get("content", "")

    if message_type == "human":
        return HumanMessage(content=content)

    elif message_type == "ai":
        return AIMessage(content=content)

    return None


def save_chat_store():
    try:
        data = {}

        for session_id, chat_history in chat_store.items():
            data[session_id] = []

            for message in chat_history.messages:
                data[session_id].append(
                    message_to_dict(message)
                )

        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        print("\nCould not save chat memory.")
        print("Error:", e)


def load_chat_store():
    if not os.path.exists(MEMORY_FILE):
        return

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError:
        print("\nMemory file is corrupted. Starting with empty memory.")
        return

    except Exception as e:
        print("\nCould not load chat memory.")
        print("Error:", e)
        return

    if not isinstance(data, dict):
        print("\nInvalid memory file format. Starting with empty memory.")
        return

    for session_id, messages in data.items():
        chat_store[session_id] = InMemoryChatMessageHistory()

        if not isinstance(messages, list):
            continue

        for message_data in messages:
            message = dict_to_message(message_data)

            if message is not None:
                chat_store[session_id].add_message(message)


def add_to_memory(session_id, user_content, ai_content):
    chat_history = get_session_history(session_id)

    chat_history.add_message(
        HumanMessage(content=user_content)
    )

    chat_history.add_message(
        AIMessage(content=ai_content)
    )

    save_chat_store()


def clear_session_memory(session_id):
    if session_id in chat_store:
        del chat_store[session_id]
        save_chat_store()
        return True

    return False