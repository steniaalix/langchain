from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough,RunnableBranch,RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from pydantic import BaseModel , Field
from typing import List

import json
import os

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)
structure_based_llm=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)


parser = StrOutputParser()


# -----------------------------
# Explanation Chain
# -----------------------------

exp_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI tutor.

Rules:
- Explain according to the student's level.
- Give a simple and short explanation.
- Be precise.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}

Explain this topic clearly.
"""
    )
])

exp_chain = exp_prompt | llm | parser


# -----------------------------
# Flashcard Structured Output
# -----------------------------

class Flashcard(BaseModel):
    question: str = Field(description="The question of the flashcard")
    answer: str = Field(description="The answer of the flashcard")


class FlashcardSet(BaseModel):
    topic: str = Field(description="The topic for the flashcards")
    explanation: str = Field(description="A short explanation on the topic")
    flashcards: List[Flashcard] = Field(description="A list of exactly three flashcards")
    quiz_question: str = Field(description="A quiz question to test understanding")


structured_llm = structure_based_llm.with_structured_output(FlashcardSet)

flashcard_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI teacher who prepares flashcards.

Rules:
- Create exactly three flashcards.
- Give a short explanation on the topic before flashcards.
- Be clear and simple.
- Match the student's level.
"""
    ),
    (
        "human",
        """
Topic: {topic}
Student level: {level}

Generate the flashcard set.
"""
    )
])

flashcard_chain = flashcard_prompt | structured_llm

#----------------------------------
#Batch Flashcards
#---------------------------------

batch_flashcards_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI teacher who prepares flashcards for all the toipcs given.

Rules:
- Create exactly three flashcards for each topic.
- Give a short explanation on the topic before flashcards.
- Be clear and simple.
- Match the student's level.
"""
    ),
    (
        "human",
        """
Topic:{topic}
Level:{level}"""
    )
])

batch_flashcards_chain=batch_flashcards_prompt|structured_llm
#------------------------------
#Quiz Topic
#------------------------------

quiz_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI tutor who creates quizes in the given topic according to the student level.
Rules:
-Make exactly three quizes
-Be clear and simple
-Give a short explanation before giving the quiz"""
    ),(
        "human",
        """
Topic:{topic}
Level:{level}"""
    )
])

quiz_chain=quiz_prompt|llm|parser

#------------------------------
#Project Chain
#------------------------------

project_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI assistant who gives projectt ideas based on the topic and level.
Rules:
-Be clear and simple
-Give exactly three project ideas
-Give it based on the user's level
-Give a short explanation of the project"""
    ),(
        "human",
        """
Topic: {topic}
Level: {level}"""
    )
])

project_chain=project_prompt|llm|parser

#-----------------------------------
#Analyzer
#------------------------------------

analogy_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI assistant who gives real-life analogy based on the topic.
Rules:
-Give real-life analogy
-Give it according to the user's level
-Be clear and simple"""
    ),(
        "human",
        """
Topic: {topic}
Level: {level}"""
    )
])

analogy_chain=analogy_prompt|llm|parser


mistake_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI assistant who gives a common mistake for a topic.
Rules:
-Give a common mistake based on topic 
-Give it based on the user's level
-Be clear and simple"""
    ),(
        "human",
        """
Topic: {topic}
Level: {level}"""
    )
])

mistake_chain=mistake_prompt|llm|parser

analyze_chain=RunnableParallel({
    "explanation":exp_chain,
    "analogy":analogy_chain,
    "mistake":mistake_chain,
    "project":project_chain,
    "quiz":quiz_chain
})
#-------------------------------
#Study pack chain .assign()
#-------------------------------

study_plan_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI study assistant. You have to create a study plan based on all given information

Rules:
-Be clear and simple
-Create it according to the user's level
-Create it based on the all give infomations
-Be friendly"""
    ),(
        "human",
        """
Topic: {topic}
Level: {level}
Explanation: {explanation}
Analogy: {analogy}
Common Mistake:{mistake}
Quiz: {quiz}
Project Ideas: {project}"""
    )
])

study_plan_chain=study_plan_prompt|llm|parser

study_pack_chain=(
    RunnablePassthrough.assign(
        explanation=exp_chain
    )
    |RunnablePassthrough.assign(
        analogy=analogy_chain
    )
    |RunnablePassthrough.assign(
        mistake=mistake_chain
    )
    |RunnablePassthrough.assign(
        quiz=quiz_chain
    )
    |RunnablePassthrough.assign(
        project=project_chain
    )
    |RunnablePassthrough.assign(
        study_plan=study_plan_chain
    )
)
#-----------------------
#Chat using memory
#-----------------------

chat_prompt=ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI study companion who remembers the conversation.

Rules:
- Remember the conversation.
- Answer follow-up questions using chat history.
- Help the student learn clearly.
- If the student asks what they said earlier, use the history.
"""
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{input}")
])

chat_chain=chat_prompt|llm

chat_store={}
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE=os.path.join(BASE_DIR,"chat_memory.json")

def get_session_history(session_id: str):
    if session_id not in chat_store:
        chat_store[session_id]=InMemoryChatMessageHistory()
    return chat_store[session_id]

def message_to_dict(message):
    return{
        "type": message.type,
        "content":message.content
    }

def dict_to_message(data):
    if not isinstance(data,dict):
        return None
    
    message_type=data.get("type")
    content=data.get("content","")
    if message_type=="human":
        return HumanMessage(content=content)
    elif message_type=="ai":
        return AIMessage(content=content)
    
    return None

def save_chat_store():
    data={}

    for session_id , chat_history in chat_store.items():
        data[session_id]=[]

        for message in chat_history.messages:
            data[session_id].append(message_to_dict(message))

    with open(MEMORY_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=4)

def load_chat_store():
    if not os.path.exists(MEMORY_FILE):
        return
    
    with open(MEMORY_FILE,"r",encoding="utf-8") as f:
        data=json.load(f)
    if not isinstance(data,dict):
        print("Invalid memory file format. Starting with empty memory.")
        return
        

    for session_id,messages in data.items():
        chat_store[session_id]=InMemoryChatMessageHistory()

        if not isinstance(messages,list):
            continue

        for message_data in messages:
            message=dict_to_message(message_data)

            if message is not None:
                chat_store[session_id].add_message(message)

memory_chatbot=RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# -----------------------------
# Default Chain
# -----------------------------

default_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI study assistant."
    ),
    (
        "human",
        """
The command is unclear.

User command: {command}

Tell the user to use:
- explain
- flashcards
- batch flashcards
- quiz
- project
- history
- analyze
- study pack
- chat
- chat history
- clear memory
- exit
"""
    )
])

default_chain = default_prompt | llm | parser


# -----------------------------
# Router Conditions
# -----------------------------

def is_exp(input_data):
    return input_data["command"] == "explain"


def is_flashcard(input_data):
    return input_data["command"] == "flashcards"

def is_quiz(input_data):
    return input_data["command"]== "quiz"

def is_project(input_data):
    return input_data["command"]== "project"

def is_analyze(input_data):
    return input_data["command"]== "analyze"

def is_study_pack(input_data):
    return input_data["command"]== "study pack"




branch_chain = RunnableBranch(
    (is_exp, exp_chain),
    (is_flashcard, flashcard_chain),
    (is_quiz,quiz_chain),
    (is_project,project_chain),
    (is_analyze,analyze_chain),
    (is_study_pack,study_pack_chain),
    default_chain
)


# -----------------------------
# Helper Function to Print Flashcards
# -----------------------------

def print_flashcards(response):
    print("\n" + "=" * 60)
    print(f"Topic: {response.topic}")
    print("=" * 60)

    print("\nExplanation:")
    print(response.explanation)

    print("\nFlashcards:")
    for index, card in enumerate(response.flashcards, start=1):
        print(f"\nFlashcard {index}")
        print(f"Q: {card.question}")
        print(f"A: {card.answer}")

    print("\nQuiz Question:")
    print(response.quiz_question)

#-------------------------------------------------
#Helper function to print batch of flashcards
#-------------------------------------------------

def print_batch_flashcards(topics,responses):
    for topic,response in zip(topics,responses):
    
        if isinstance(response,Exception):
            print("Failed to generate flashcards for this topic.")
            print(response)
            continue

        print_flashcards(response)
        print("\n"+"-"*70)
#-----------------------------------
#Helper function to print analyze
#-----------------------------------

def print_analyzer(response):
    print("\n" + "=" * 70)
    print("AI TOPIC ANALYZER")
    print("=" * 70)

    print("\nExplanation:")
    print(response["explanation"])

    print("\nAnalogy:")
    print(response["analogy"])

    print("\nCommon Mistake:")
    print(response["mistake"])

    print("\nProject Ideas:")
    print(response["project"])

    print("\nQuiz:")
    print(response["quiz"])
#--------------------------------------
#Helper funciton to print study pack
#--------------------------------------

def print_study_pack(response):
    print("\n" + "=" * 70)
    print("AI STUDY PACK")
    print("=" * 70)

    print("\nTopic:")
    print(response["topic"])

    print("\nLevel:")
    print(response["level"])

    print("\nExplanation:")
    print(response["explanation"])

    print("\nAnalogy:")
    print(response["analogy"])

    print("\nCommon Mistake:")
    print(response["mistake"])

    print("\nQuiz:")
    print(response["quiz"])

    print("\nProject Ideas:")
    print(response["project"])

    print("\nStudy Plan:")
    print(response["study_plan"])
#------------------------------------------------
#Helper to convert any type of response to text
#------------------------------------------------

def convert_to_text(command,response):
    if isinstance(response,FlashcardSet):
        text=f"\n\nTopic: {response.topic}\n\n"
        text+=f"Explanation: {response.explanation}\n\n"
        text+=f"Flashcards:\n"

        for index,card in enumerate(response.flashcards,start=1):
            text+= f"{index}. Q: {card.question}\n"
            text+= f"  A: {card.answer}\n"

        text+= f"\nQuiz Question: {response.quiz_question}"
        return text
    
    if isinstance(response,dict):
        text=""

        for key,value in response.items():
            text+= f"\n{key.upper()}:\n{value}\n"

        return text
    
    return str(response)

#----------------------------------------------------------
#Converting batch flashcard response into text format
#----------------------------------------------------------

def batch_response_to_text(topics,responses):

    text="Batch Flashcards Result\n\n"

    for topic,response in zip(topics,responses):
        text+= "="*50+"\n"
        text+= f"Topic: {topic.strip()}\n"
        text+= "="*50+"\n"
    
        if isinstance(response,Exception):
            text+= f"Failed to generate flashcards: {response}\n\n"
            continue

        text+= f"Explanation: {response.explanation}\n\n"
        text+= "flashcards:\n"

        for index,card in enumerate(response.flashcards,start=1):
            text+= f"{index}. Q: {card.question}\n"
            text+= f"   A:{card.answer}\n"
    
        text+= f"\nQuiz Question: {response.quiz_question}\n\n"
    
    return text

# -----------------------------
# Main App
# -----------------------------

history=[]
load_chat_store()
while True:
    command = input("\nEnter command explain/flashcards/Batch Flashcards/Quiz/Project/History/Analyze/Study Pack/Chat/Chat History/Clear Memory/exit: ").strip().lower()
    if command == "exit":
        save_chat_store()
        print("Bye!")
        break
    
    
    if command not in ["exit","history"]:
        session_id=input("Enter session ID:").strip().lower()


    if command == "clear memory":
        if session_id in chat_store:
            del chat_store[session_id]
            save_chat_store()
            print(f"Memory cleared for session: {session_id}")
        else:
            print(f"No memory was found for session: {session_id}")
        
        continue
    
    if command=="batch flashcards":
        topic_text=input("Enter the topics seperated by commas:").strip()
        topics=topic_text.split(",")
        level=input("Enter level:").strip().lower()

        input_batch=[]

        for topic in topics:
            input_batch.append({
                "topic":topic.strip(),
                "level":level
            })

        
        responses=batch_flashcards_chain.batch(input_batch,return_exceptions=True)

        print_batch_flashcards(topics,responses)
        history.append({
            "command":command,
            "topic":topic_text,
            "level":level
        })
        
        batch_text=batch_response_to_text(topics,responses)

        chat_history=get_session_history(session_id)
        chat_history.add_message(
            HumanMessage(
                content=f"\n Command: {command}\n Topics: {topic_text}\n Level: {level}"
            )
        )

        chat_history.add_message(
            AIMessage(
                content=batch_text
            )
        )
        save_chat_store()
        continue



    if command=="chat":
        config={
                "configurable":{"session_id":session_id}
            }
        while True:
            
            user_input=input(f"\nYou:")
            if user_input.strip().lower() in ["exit","quit"]:
                print("Leaving chat mode.....")
                break
            response=memory_chatbot.invoke({
                "input":user_input
            },config=config
            )

            print("\nAI:")
            print(response.content)
            save_chat_store()
        continue

    if command=="chat history":
        chat_history=get_session_history(session_id)
        if not chat_history.messages:
            print("\nNo chat history yet...")
        else:
            print("\nChat Memory:")
            for message in chat_history.messages:
                print(type(message).__name__,":",message.content)
        continue


    if command=="history":
        if not history:
            print("\nNo history yet!!")
        else:
            for index,item in enumerate(history,start=1):
                print(f"{index}:-")
                print(f"Command: {item['command']}")
                print(f"Topic: {item['topic']}")
                print(f"Level: {item['level']}")
        continue
    topic = input("Enter the topic: ").strip()
    level = input("Enter the level: ").strip()

    input_data = {
        "command": command,
        "topic": topic,
        "level": level
    }
    history.append(input_data)
    response = branch_chain.invoke(input_data)

    if isinstance(response, FlashcardSet):
        print_flashcards(response)
    
    elif command=="analyze":
        print_analyzer(response)
    
    elif command=="study pack":
        print_study_pack(response)
    else:
        print("\nAI:")
        print(response)

    response_text=convert_to_text(command,response)

    chat_history=get_session_history(session_id)

    chat_history.add_message(
        HumanMessage(
            content=f"Command: {command}\n Topic: {topic}\nLevel: {level}"
        )
    )

    chat_history.add_message(
        AIMessage(
            content=response_text
        )
    )
    save_chat_store()


    print("\n" + "-" * 60)
