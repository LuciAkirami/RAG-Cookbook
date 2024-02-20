# loading the API Key
from dotenv import load_dotenv
import os

load_dotenv()

"""## Multi-Query"""
"""
Langchain's MultiQueryRetriever

The MultiQueryRetriever automates the process of prompt tuning by using an LLM to generate multiple \
queries from different perspectives for a given user input query. For each query, it retrieves a \
set of relevant documents and takes the unique union across all queries(de-duplication) to get a larger set of \
otentially relevant documents.
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

# Loading a single website
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

# Load the documents from the website
docs = loader.load()

# Initialize a RecursiveCharacterTextSplitter instance
# This will split the documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# Split the documents into chunks
chunks = text_splitter.split_documents(docs)

# Initialize a GoogleGenerativeAIEmbeddings instance
# This will be used to embed the chunks into vectors
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a Chroma vectorstore from the documents and embeddings
# This will store the vectors for efficient retrieval and comparison
vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)

# Initiating the Language Model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Providing the User Prompt
question = "What are the approaches to Task Decomposition?"

# Creating an instance of the MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

# Performing Multi Query Retreival and obtaining the Unique Documents
unique_docs = retriever_from_llm.get_relevant_documents(query=question)

# To print the length of unique documents retreived 
# print(len(unique_docs))

# Creating the Prompt Template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Passing the PROMT along with the Unique Documents Retrieved from MultiQueryRetriever
response = llm.invoke(input=PROMPT.format_prompt(
    context=unique_docs,
    question=question
).text)

# Printing the Response
print(response)