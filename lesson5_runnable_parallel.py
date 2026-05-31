from langchain_core.runnables import RunnableParallel,RunnableLambda

def make_uppercase(text):
    return text.upper()

def count_character(text):
    return len(text)

parallel_chain=RunnableParallel({
    "uppercase": RunnableLambda(make_uppercase),
    "length":RunnableLambda(count_character)
})

response=parallel_chain.invoke("langchain")
print(response)