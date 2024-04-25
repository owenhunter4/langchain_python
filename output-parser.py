from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser,JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel,Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7)

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a content creator. Write me a tweet"),
        ("human","tell me a joke about {input}")
    ])

    parser = StrOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input":"duck"
    })

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a AI"),
        ("human","Generate a list of 10 sysnonyms for the {input}. Return the result as a comma seperated list.")
    ])

    parser = CommaSeparatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({
        "input":"happy"
    })

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system","Do following command below"),
        ("human","Extract information from the following phrase '{phrase}'.\nFormatting Instrctions:{format_instructions}")
    ])

    class Person(BaseModel):
        name: str = Field(description="the name of the person")
        age: int = Field(description="the age of the person")
        address: str = Field(description="the address of the person")

    class Food(BaseModel):
        recipe: str=Field(description="the name of the recipe")
        ingredients: list=Field(description="ingredients")


    # parser = JsonOutputParser(pydantic_object=Person)
    parser = JsonOutputParser(pydantic_object=Food)

    chain = prompt | model | parser

    # return chain.invoke({
    #     "phrase":"Owen is 40 year old. He living at 222 Avanue road Thailand",
    #     "format_instructions":parser.get_format_instructions()
    # })
    return chain.invoke({
        "phrase":"the ingredients for KFC are chicken, oil, wine, onions, cheese",
        "format_instructions":parser.get_format_instructions()
    })

# print(call_string_output_parser())
# print(call_list_output_parser())
print(call_json_output_parser())