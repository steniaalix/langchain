from langchain_core.runnables import RunnablePassthrough
'''
chain=RunnablePassthrough.assign(topic_length=lambda x: len(x["topic"]))

response=chain.invoke({
    "topic":"Langchain"
})

print(response)
'''

chain=RunnablePassthrough.assign(
    topic_length=lambda x: len(x["topic"]),
    uppercase_topic=lambda x: x["topic"].upper()
)
response=chain.invoke({
    "topic":"python functions"
})

print(response)