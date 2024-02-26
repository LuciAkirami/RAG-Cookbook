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


# ---------------------------------------------- Method 3 ----------------------------------------------
"""## Step Back Prompting """
"""
The prompting technique called "Step-Back" prompting can improve performance on complex questions by 
first asking a "step back" question. This can be combined with regular question-answering applications 
by then doing retrieval on both the original and step-back question. The Step Back Prompting helps in both 
RAG and Non-RAG context

Eg:
You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.

Here are a few examples:

Original Question: Which position did Knox Cunningham hold from May 1955 to Apr 1956? 
Stepback Question: Which positions have Knox Cunning- ham held in his career?

Original Question: Who was the spouse of Anna Karina from 1968 to 1974? S
tepback Question: Who were the spouses of Anna Karina?

Original Question: Which team did Thierry Audel play for from 2007 to 2008? 
Stepback Question: Which teams did Thierry Audel play for in his career?
"""


# Creating a function for the search tool
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def retriever(query):
    return search.run(query)


# Creating a prompt template for assigning few shot examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Below are the few shot examples
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]

# Defining our ChatPrompt Template
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Finally we pass the Prompt Template with the examples to the FewShotChatMessagePromptTempate
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# print(few_shot_prompt)


# Creating the final Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

# print(prompt.messages)


# Creating the Model object
from langchain_google_genai import ChatGoogleGenerativeAI

# SystemMessages are not yet supported in ChatGoogleGenerativeAI, hence set convert_system_message_to_human to True, 
# to automatically convert the leading SystemMessage to a HumanMessage,
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)


# Lets generate some questions
# defining our step_back question generation chain with LCEL
step_back_qn_gen = prompt | llm | StrOutputParser()

question = "was chatgpt around while trump was president?"

step_back_question = step_back_qn_gen.invoke({"question": question})

print("Actual Question:",question)
print("Step Back Question:",step_back_question)


# Checking the Retrieval/Context without Step Back
retrieved_data = retriever(question)
print("Retrieved Data - Without Step Back:\n",retrieved_data)

# Checking the Retrieval/Context with Step Back
retrieved_data = retriever(step_back_qn_gen.invoke({"question": question}))
print("Retrieved Data - with Step Back:\n",retrieved_data)


# Creating the Final Prompt that takes in both Original and Step Back Context
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{normal_context}
{step_back_context}

Original Question: {question}
Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# Uncomment and Run the below if you want the prompt from langchainhub instead of the above
# from langchain import hub

# response_prompt = hub.pull("langchain-ai/stepback-answer")


# Creating our Final Chain
from langchain_core.runnables import RunnableLambda

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": step_back_qn_gen | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)


# Testing the previous prompt with the Step Back Method
step_back_approach_response = chain.invoke({"question": question})
print("Answer with Step Back Approach:\n",step_back_approach_response)


# Testing the BaseLine Approach without Step Back
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{normal_context}

Original Question: {question}
Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | llm
    | StrOutputParser()
)

without_stepback_response = chain.invoke({"question": question})
print("Answer without Step Back:\n",without_stepback_response)

# ---------------------------------------------- Method 4 ----------------------------------------------
"""## HyDE - Hypothetical Document Embeddings"""
"""
HyDE uses Language Model to provide a "Hypothetical" response before searching for a match in the embeddings. 
In contrast to the query to answer embedding similar search used in the conventional RAG retrieval approach, we 
are performing answer to answer embedding similarity search here.

For example, let's say a user asks what's the best item at KFC. Here the question talks about a food item, due the 
absence of the specific food item in the query, it becomes problematic to perform a RAG here

Nevertheless, this strategy has a flaw in that it could not always yield positive outcomes. For example, this strategy 
is ineffective and may result in more inaccurate information being generated if the topic being addressed is completely 
unknown to the language model.

It works because the Hypothetical Document (hypothetical answer generated by LLM) will be closer to the desired doc in 
the vector embedding space than the sparse raw input questions
"""

# --- Model Loading ---
# Import the necessary modules from the langchain_google_genai package.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Create a ChatGoogleGenerativeAI object, specifying the model to use and whether to convert system messages to 
# human-readable format.
llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

# Create a GoogleGenerativeAIEmbeddings object for embedding our prompts and documents
base_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# --- Data Loading ---
# Import the WebBaseLoader class from the langchain_community.document_loaders module.
from langchain_community.document_loaders import WebBaseLoader

# Create a WebBaseLoader object with the URL of the blog post to load.
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")

# Load the blog post and store the documents in the `docs` variable.
docs = loader.load()


# --- Splitting / Creating Chunks ---
# Import the RecursiveCharacterTextSplitter class from the langchain.text_splitter module.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a RecursiveCharacterTextSplitter object using the provided chunk size and overlap.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300,
                                                                     chunk_overlap=50)

# Split the documents in the `docs` variable into smaller chunks and store the resulting splits in the `splits` variable.
splits = text_splitter.split_documents(docs)


# --- Creating the HyDe Embedder ---
import langchain
from langchain.chains import HypotheticalDocumentEmbedder

# Create a HypotheticalDocumentEmbedder object
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(llm,
                                                   base_embeddings,
                                                   prompt_key = "web_search")

# Print the prompt used by the LLM chain.
print("HyDE Embedding Prompt")
print(hyde_embeddings.llm_chain.prompt)

# Set True for debugging to view the logs
langchain.debug = True


# --- HyDE Embedding Example ---
# Now we can use it as any embedding class!
result = hyde_embeddings.embed_query("Where is Taj Mahal?")

# the result is an embedding
print("HyDE Embedding Example")
print(result)


# --- Creating Embeddings by Passing Hyde Embeddings to Vector Store ---
from langchain_community.vectorstores import Chroma

# passing the hyde embeddings to create and store embeddings
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=hyde_embeddings)

# Creating Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


# --- Creating the Final RAG ---
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Creating the Prompt Template
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Creating a function to format the retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Optional - If you want go with RAG Prompt from LangchainHub
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")
print('RAG Response with HyDE')
print(response)

# Optional
# Creating RetrievalQA Chain
# from langchain.chains import RetrievalQA

# # Creating the retriever from vectorstore
# retriever = vectorstore.as_retriever()

# qa = RetrievalQA.from_chain_type(llm=llm,
#                                  chain_type="stuff",
#                                  retriever=retriever)

# query = "What is Task Decomposition?"
# response = qa.invoke(input=query)
# print(response['result'])