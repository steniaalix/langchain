from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
'''
prompt=ChatPromptTemplate.from_template("Explain {topic} in a simple way.")
result=prompt.invoke({"topic":"LangChain"})
print(result)

prompt=ChatPromptTemplate.from_template("Explain {topic} to a {level} student in {style} style.")
result=prompt.invoke({"topic":"APIs","level":"beginner","style":"friendly"})
print(result)


prompt=ChatPromptTemplate.from_messages([("system","You are a helpful AI tutor. Explain things clearly."),("human","Explain {topic} with an eample.")])

result=prompt.invoke({"topic":"Python functions"})
print(result)
'''
