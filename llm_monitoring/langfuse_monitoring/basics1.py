"""
To monitor LLMs in Langchain, we can define a Langfuse Callback Handler and then
pass this handler to the invoke function of chain. So when the chain is run, all the
intermediate steps along with input and output gets logged in the Langfuse and we can
view it in the Langfuse dashboard, i.e. the sequence of steps from start to finish
"""

import os
from dotenv import load_dotenv

load_dotenv()

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langfuse.callback import CallbackHandler

# creating our langfuse callback
langfuse_handler = CallbackHandler(
    public_key=os.environ["PUBLIC_KEY"],
    secret_key=os.environ["SECRET_KEY"],
    host="http://localhost:3000",
)

# creating two prompt templates
prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

# instantiating our model
model = ChatOpenAI()

# creating our chains
chain1 = prompt1 | model | StrOutputParser()
chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

# invoking the chain with langfuse_handler
chain2.invoke(
    {"person": "obama", "language": "spanish"}, config={"callbacks": [langfuse_handler]}
)
