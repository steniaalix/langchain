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