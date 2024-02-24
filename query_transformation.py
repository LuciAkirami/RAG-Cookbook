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