import os
from dotenv import load_dotenv


load_dotenv()


NORMAL_MODEL = os.getenv("NORMAL_MODEL", "openai/gpt-oss-120b")
STRUCTURED_MODEL = os.getenv("STRUCTURED_MODEL", "openai/gpt-oss-120b")

NORMAL_TEMPERATURE = float(os.getenv("NORMAL_TEMPERATURE", "0.7"))
STRUCTURED_TEMPERATURE = float(os.getenv("STRUCTURED_TEMPERATURE", "0.1"))

MEMORY_FILE_NAME = os.getenv("MEMORY_FILE_NAME", "chat_memory.json")

LANGSMITH_PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "ai-study-companion")