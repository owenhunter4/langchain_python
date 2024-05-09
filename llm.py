import os
# from langchain_openai import ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY,temperature=0.3)
llm = ChatOpenAI(api_key="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")

# tweet_prompt = ChatPromptTemplate.from_template("You are a content creator. Write me a tweet about {topic}.")
tweet_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a content creator. Write me a tweet"),
    ("human", "{input}")
])

tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)

if __name__ == "__main__":
    topic = "how ai is really cool"
    # resp = tweet_chain.invoke(topic)
    resp = tweet_chain.invoke({"input": topic})
    print(resp)
