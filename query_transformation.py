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




# ---------------------------------------------- Method 2 ----------------------------------------------
"""## Rewrite-Retrieve-Read """
"""
Because the original query can not be always optimal to retrieve for the LLM, especially in the real world... 
we first prompt an LLM to rewrite the queries, then conduct retrieval-augmented reading.
"""

# Creating a function for the search tool
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def retriever(query):
    return search.run(query)


# Creating the Chat Prompt Tempalte
from langchain_core.prompts import ChatPromptTemplate

template = """Answer the users question if and only if its present in the following context:

<context>
{context}
</context>

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Instatiating our model
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")


# Creating our chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Testing the chain - Baseline Response
simple_query = "what is langchain?"

resopnse = chain.invoke(simple_query)

print("Baseline Response: ",response)


# Creating a Distracted Query
distracted_query = "man that sam bankman fried trial was crazy! what is langchain?"

resopnse = chain.invoke(distracted_query)

print("Baseline Response for Distracted Query: ",response)


# Reason for Bad Response - Distracted Query Results in Bad Retrieval
retrieved_content = retriever(distracted_query)
print("Retrieved Information from Distracted Query")
print(retrieved_content)


# Creating a prompt template to rewrite the prompt so that the search will give better results

template = """Provide a better search query for \
web search engine to answer the given question, end \
the queries with ’**’. Question: \
{x} Answer:"""

rewrite_prompt = ChatPromptTemplate.from_template(template)
# rewrite_prompt.messages[0].prompt.template


# Instead of creating the above prompt, the langchainhub already contains a pre-created prompts
# So to use below, instead of above, uncomment the below
# Here the rewrite_prompt contains the same prompt as the one written above

# from langchain import hub

# rewrite_prompt = hub.pull("langchain-ai/rewrite")
# print(rewrite_prompt)


# Parser to remove the `**`

def _parse(text):
    return text.strip("**")


# Creating the rewriter
# The rewriter will take in the user question, put it in the rewrite_prompt to rewrite it
# and then parses it to remove the **

rewriter = rewrite_prompt | llm | StrOutputParser() | _parse


# Putting them all together - Using LCEL (Langchain Expression Language)

rewrite_retrieve_read_chain = (
    {
        "context": {"x": RunnablePassthrough()} | rewriter | retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Testing the distracted prompt using Rewrite-Retrieve-Read method

response = rewrite_retrieve_read_chain.invoke(distracted_query)

print("Rewrite-Retrieve-Read Response for Distracted Query: ",response)