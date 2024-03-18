from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from langchain_openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# loading the langfuse documents
loader = WebBaseLoader("https://langfuse.com/docs")
docs = loader.load()

# creating chunks out of them
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)

# storing the documents in vecstore
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# making the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# ----------------- Creating Tools ---------------------------
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

# search tool
search_tool = TavilySearchResults()

# retriever tool
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="langfuse_search",
    description="Search for information about Langfuse For any questions on Langfuse, you must use this tool",
)

tools = [search_tool, retriever_tool]

# ---------------- Creating the Agent ---------------------------
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[], template="You are a helpful assistant"
            )
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["input"], template="{input}")
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, tools, prompt)

# ------------------------- Creating Agent Executor ---------------------------------
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ------------------------- Executing Agents ----------------------------------
from rich import print

if __name__ == "__main__":
    # agent not using any tools
    res = agent_executor.invoke({"input": "hi!"})
    print(res)

    # agent using retriever tool
    res = agent_executor.invoke({"input": "What is Langfuse?"})
    print(res)

    # agent using search tool
    res = agent_executor.invoke({"input": "whats the weather in sf?"})
    # output is present in res['output']
    print(res["output"])
