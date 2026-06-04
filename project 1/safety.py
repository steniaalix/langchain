#------------------------------------------------------
#Helper for naormal chains for Exception handling
#------------------------------------------------------

def safe_invoke(chain, input_data):
    try:
        return chain.invoke(input_data)
    
    except Exception as e:
        print("\nSomething went wrong while generating the response.")
        print("Error:",e)
        return None
    

#-------------------------------------------------------
#Helper for chat memory calls Exception handling
#-------------------------------------------------------

def safe_memory_invoke(memory_chatbot,user_input,config):
    try:
        return memory_chatbot.invoke(
            {
                "input": user_input
            },
            config=config

        )

    except Exception as e:

        print("\nSomething went wrong in chat mode.")
        print("Error:",e)
        return None

#-------------------------------------------------------
# #Helper for batch flashcards Exception handling
#-------------------------------------------------------

def safe_batch(chain, input_batch):
    try:
        return chain.batch(
            input_batch,
            return_exceptions=True
        )
    
    except Exception as e:
        print("\nSomething went wrong while generating batch flashcards.")
        print("Error:",e)
        return None
#-------------------------------------------------
#Helper for empty input Excetion
#-------------------------------------------------

def get_non_empty_input(prompt_text):


    while True:
        value= input(prompt_text).strip()

        if value:
            return value
        
        print("Input can not be empty. Please try again.")


#------------------------------------------
#Helper for stream
#------------------------------------------

def safe_stream(chain,input_data):
    try:
        full_response=""

        for chunk in chain.stream(input_data):
            print(chunk,end="",flush=True)
            full_response+=chunk
        
        print()
        return full_response
    
    except Exception as e:
        print("\nSometthing went wrong while streaming the response.")
        print("Error:",e)
        return None