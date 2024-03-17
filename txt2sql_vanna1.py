from vanna.flask import VannaFlaskApp
from vanna.remote import VannaDefault
from vanna.vannadb.vannadb_vector import VannaDB_VectorStore
from vanna.ollama import Ollama

import os
from dotenv import load_dotenv

load_dotenv()

# by default it uses OpenAI GPT 3.5 from Vanna to generate SQL
vn = VannaDefault(model='chinook', api_key=os.environ['VANNA_API_KEY'])

# connecting to a database hosted online on Vanna
vn.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')

# try this in colab with visualize = True
vn.ask("What are the top 10 albums by sales?", visualize=False)

# -------------------------- Using Open Source Models --------------------------

# create a custom MyVanna Class with Ollama
# here we can use other VectorStore instead Vanna Hosted VectorStore
# to use Vanna hosted VectorStore, we pass in the API Key
class MyVanna(VannaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        VannaDB_VectorStore.__init__(self, vanna_model='chinook', vanna_api_key=os.environ['VANNA_API_KEY'], config=config)
        Ollama.__init__(self, config={'model': 'phi'}) 

vn = MyVanna()
vn.connect_to_sqlite('https://vanna.ai/Chinook.sqlite')
vn.ask("What are the top 10 albums by sales?")

# ----------------------- Using Inbuilt Vanna Flask App ---------------------------

# setting allow_llm_see_data = True will let us ask follow up questions and summarise the 
# results generated after running the SQL query
app = VannaFlaskApp(vn,allow_llm_to_see_data=True)
app.run()

