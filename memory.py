from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory

from dotenv import load_dotenv
load_dotenv()

UPSTASH_URL = "https://apn1-light-sheepdog-34699.upstash.io"
UPSTASH_TOKEN = "AYeLASQgOGJlZWZjM2YtMTM1ZS00MDNhLWFkYmItM2NhNzE2OGNiNDA4MTg5NGU1MjVlNzI3NDM3ZDlmYTU0OTEzNDdkMjcxYzY="


history = UpstashRedisChatMessageHistory(
    url=UPSTASH_URL,
    token=UPSTASH_TOKEN,
    session_id="chat1"
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
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


# msg1 = {
#     "input": "what's the most bueatyful sea of Thailand?"
# }

# response1 = chain.invoke(msg1)
# print(response1)

msg2 = {
    "input": "is there hot?"
}

response2 = chain.invoke(msg2)
print(response2)
