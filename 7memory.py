import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

from dotenv import load_dotenv
load_dotenv()

# UPSTASH_URL = "https://apn1-light-sheepdog-34699.upstash.io"
# UPSTASH_TOKEN = "AYeLASQgOGJlZWZjM2YtMTM1ZS00MDNhLWFkYmItM2NhNzE2OGNiNDA4MTg5NGU1MjVlNzI3NDM3ZDlmYTU0OTEzNDdkMjcxYzY="


history = UpstashRedisChatMessageHistory(
    url=os.environ['UPSTASH_URL'],
    token=os.environ['UPSTASH_TOKEN'],
    session_id="chat1"
)

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

# chain =  prompt | model
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        msg = {
            "input": user_input
        }
        response = chain.invoke(msg)
        print("AI: "+response["text"])
