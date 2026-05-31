'''
from langchain_core.runnables import RunnablePassthrough

chain=RunnablePassthrough()

response=chain.invoke("hello Langchain")

print(response)
'''

from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda

def uppercase(text):
    return text.upper()

def count_letters(text):
    return len(text)

chain=RunnableParallel({
    "original":RunnablePassthrough(),
    "uppercase":RunnableLambda(uppercase),
    "length":RunnableLambda(count_letters)
})
response=chain.invoke("langchain")

print(response)