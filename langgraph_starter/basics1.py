from dotenv import load_dotenv
from langgraph.graph import Graph
from langchain_google_genai.llms import GoogleGenerativeAI

load_dotenv()

# Defin model
llm = GoogleGenerativeAI(model="gemini-pro")
# Define functions
def func1(input):
    message = llm.invoke(input)
    return message

def func2(message):
    return "AI Said: "+message

# Define a Langchain Graph
workflow = Graph()

# Adding Nodes to Graph
workflow.add_node('node_1',func1)
workflow.add_node('node_2',func2)

# Defining Edges of Graph
workflow.add_edge('node_1','node_2')

# Adding entry and exit nodes
workflow.set_entry_point('node_1')
workflow.set_finish_point('node_2')

# Compilikng the Graph
app = workflow.compile()

# Running the Graph
response = app.invoke("Hello")

print(response)
