"""
------- Semantic Router in a Nutshell --------

Semantic Router is a superfast decision-making layer for your LLMs and agents. 
Rather than waiting for slow LLM generations to make tool-use decisions, we use the magic 
of semantic vector space to make those decisions — routing our requests using semantic meaning.
"""

import os
from dotenv import load_dotenv
from semantic_router import Route

# loading openai api key
load_dotenv()

# We start by defining a dictionary mapping routes to example phrases that should trigger those routes.
# All the below routes are static routes. The static routes return a Route.name when chosen, 
# whereas dynamic routes use an LLM call to produce parameter input values, i.e. for function calling
education = Route(
    name="education",
    utterances=[
        "What do you think about the current state of education?",
        "How important do you believe education is for society?",
        "Do you think the education system needs reform?",
        "Online learning seems to be gaining popularity, what are your thoughts?",
        "Teachers play a crucial role in shaping the future, don't you agree?",
        "What subjects do you think should be emphasized more in schools?",
    ],
)

sports = Route(
    name="sports",
    utterances=[
        "Did you catch the game last night?",
        "Who do you think will win the championship this year?",
        "I can't believe they lost again!",
        "That was an amazing play!",
        "Sports bring people together, don't they?",
        "I'm a huge fan of basketball, how about you?",
    ],
)


technology = Route(
    name="technology",
    utterances=[
        "Have you seen the latest smartphone release?",
        "What's your favorite gadget?",
        "Technology is advancing at an incredible pace, isn't it?",
        "Do you think AI will change the world?",
        "I'm excited about virtual reality technology, are you?",
        "How do you feel about privacy concerns related to technology?",
    ],
)


# we then initialize an embedding model
from semantic_router.encoders import OpenAIEncoder

encoder = OpenAIEncoder()

# next we deifine the route layer
from semantic_router.layer import RouteLayer

# the route layer will consume a text(a query) and output the corresponding route
# to initialize it we need to pass in the encoder and the list of routes
# By default the RouteLayer will use the OpenAI() llm 
rl = RouteLayer(
    encoder=encoder,
    routes=[education, sports, technology],
)

# testing 
response = rl('What subjects do you think should be emphasized more in schools?')
print("Query: What subjects do you think should be emphasized more in schools?")
print("Response: ", response)
print(f"Route: {response.name}",end="\n\n")

response = rl('Which smartphone is better than iPhone 12?')
print("Query: Which smartphone is better than iPhone 12?")
print("Response: ", response)
print(f"Route: {response.name}",end="\n\n")

# failed to find a route for this query
response = rl('Which sports team won the World Cup in 2020?')
print("Query: Which sports team won the World Cup in 2020?")
print("Response: ", response)
print(f"Route: {response.name}",end="\n\n")

# unrelated query
response = rl('Who is the President of France?')
print("Query: Who is the President of France?")
print("Response: ", response)
print(f"Route: {response.name}",end="\n\n")