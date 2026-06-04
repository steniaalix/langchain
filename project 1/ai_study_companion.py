
from schemas import FlashcardSet
from safety import safe_invoke,safe_memory_invoke,safe_batch,get_non_empty_input,safe_stream
from printers import print_analyzer,print_batch_flashcards,print_flashcards,print_study_pack
from memory_manager import (
    get_session_history,
    save_chat_store,
    load_chat_store,
    add_to_memory,
    clear_session_memory
)
from chains import (
    branch_chain,
    batch_flashcards_chain,
    memory_chatbot
)







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
STREAMING_COMMANDS=["explain","quiz","project"]
load_chat_store()


while True:
    command = input("\nEnter command explain/flashcards/Batch Flashcards/Quiz/Project/History/Analyze/Study Pack/Chat/Chat History/Clear Memory/exit: ").strip().lower()
    if command == "exit":
        save_chat_store()
        print("Bye!")
        break
    
    
    if command not in ["exit","history"]:
        session_id=get_non_empty_input("Enter session ID:").lower()

    if command == "clear memory":
        memory_cleared= clear_session_memory(session_id)
        if memory_cleared:
            print(f"Memory cleared for session: {session_id}")
        
        else: 
            print(f"No memory was found for session: {session_id}")


        continue
    
    if command=="batch flashcards":
        topic_text=get_non_empty_input("Enter the topics seperated by commas:")
        topics=topic_text.split(",")
        level=get_non_empty_input("enter the level").lower()

        input_batch=[]

        for topic in topics:
            input_batch.append({
                "topic":topic.strip(),
                "level":level
            })

        
        responses=safe_batch(batch_flashcards_chain,input_batch)
        if responses is None:
            continue

        print_batch_flashcards(topics,responses)
        history.append({
            "command":command,
            "topic":topic_text,
            "level":level
        })
        
        batch_text=batch_response_to_text(topics,responses)

        add_to_memory(
            session_id,
            f"Command: {command}\nTopics: {topic_text}\nLevel: {level}",
            batch_text
        )
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
            response=safe_memory_invoke(memory_chatbot, user_input, config)
            if response is None:
                continue

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
    
    
    
    
    
    topic = get_non_empty_input("Enter the topic: ")
    level = get_non_empty_input("Enter the level: ")

    input_data = {
        "command": command,
        "topic": topic,
        "level": level
    }


    history.append(input_data)


    if command in STREAMING_COMMANDS:
        print("\nAI:")
        response=safe_stream(branch_chain,input_data)
    else:
        reponse=safe_invoke(branch_chain,input_data)
    

    if response is None:
        continue


    if command in STREAMING_COMMANDS:
        pass

    elif isinstance(response,FlashcardSet):
        print_flashcards(response)

    elif command== "analyze":
        print_analyzer(response)

    elif command=="Study pack":
        print_study_pack(response)

    else:
        print("\nAI:")
        print(response)


    response_text=convert_to_text(command,response)


    add_to_memory(
        session_id,
        f"Command: {command}\nTopic: {topic}\nLevel: {level}",
        response_text
    )


    print("\n" + "-" * 60)
