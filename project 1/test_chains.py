from chains import branch_chain, batch_flashcards_chain

from schemas import FlashcardSet

def test_explain():
    input_data={
        "command":"explain",
        "topic": "Python OOP",
        "level": "beginner"
    }

    response=branch_chain.invoke(input_data)

    print("\nEXPLAIN TEST:")
    print(response)

def test_flashcards():
    input_data={
        "command": "flashcards",
        "topic": "Python lists",
        "level": "beginner"
    }

    response=branch_chain.invoke(input_data)
    print("\nFLASHCARD TEST:")

    if isinstance(response,FlashcardSet):
        print("Structured output worked.")
        print("Topic:",response.topic)
        print("Number of flashcards:",len(response.flashcards))

    else:
        print("Structured output failed.")
        print(response)


def test_analyze():
    input_data={
        "command": "analyze",
        "topic": "python functions",
        "level": "beginner"
    }

    response=branch_chain.invoke(input_data)

    print("\nANALYZE TEST:")
    print(response.keys())


def test_batch_flashcards():
    input=[
        {
            "topic": "Python lists",
            "level": "beginner"
        },
        {
            "topic": "Python tuples",
            "level": "beginner"
        }
    ]

    responses=batch_flashcards_chain.batch(
        input,
        return_exceptions=True
    )

    print("\nBATCH FLASHCARD TEST:")

    for response in responses:
        if isinstance(response, Exception):
            print("Failed:",response)
        
        elif isinstance(response, FlashcardSet):
            print("Success:",response.topic)
        
        else:
            print("Unexpected response:",response)


if __name__=="__main__":
    test_explain()
    test_flashcards()
    test_analyze()
    test_batch_flashcards()