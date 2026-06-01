from langchain_core.runnables import RunnableBranch , RunnableLambda

def is_quiz_request(user_input):
    return "quiz" in user_input.lower()

def is_project_request(user_input):
    return "project" in user_input.lower()

quiz_chain=RunnableLambda(lambda x: "This should go to the quiz chain.")
project_chain=RunnableLambda(lambda x: "This should go to the project chain")
default_chain=RunnableLambda(lambda x: "This should go to the default explanation chain.")

branch_chain=RunnableBranch(
    (is_quiz_request,quiz_chain),
    (is_project_request,project_chain),
    default_chain
)
response=branch_chain.invoke("Give me a project idea for python oop")

print(response)