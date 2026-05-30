from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate 
from dotenv import load_dotenv

load_dotenv()


def generate_pet_name():
    llm =ChatGroq(model="llama-3.3-70b-versatile")
    name=llm.invoke("generate me an image of a dog",temperature=0.6)
    return name.content
if __name__=="__main__":
    print(generate_pet_name())