from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

def get_ducuments_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    print(len(splitDocs))
    return splitDocs

def create_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorStore = FAISS.from_documents(documents=docs,embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.2)

    prompt = ChatPromptTemplate.from_template("""
                                            Answer the user's question:
                                            Context:{context}
                                            Question: {input}
                                            """)

    chain = create_stuff_documents_chain(
        llm=model, prompt=prompt
    )
    retriever = vectorStore.as_retriever(search_kwargs={"k":1})

    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retrieval_chain

docs = get_ducuments_from_web("https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/")
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input" : "How to Install Libraries in a Virtual Environment?",\
})

# print("context ",response["context"])
print(response["answer"])
bk