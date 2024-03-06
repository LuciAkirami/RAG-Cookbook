from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder
from dotenv import load_dotenv

load_dotenv()

# Creating Routes
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

# initializing the encoder
encoder = OpenAIEncoder()

# adding the Routes to the RouteLayer
rl = RouteLayer(
    routes=[education, sports],
    encoder=encoder,
)

# testing 
response = rl('What subjects do you think should be emphasized more in schools?')
print("Query: What subjects do you think should be emphasized more in schools?")
print("Response: ", response)
print(f"Route: {response.name}",end="\n\n")

# -------------------- Saving Routes --------------------- #
# can save to yaml too
rl.to_json('layer.json')

# -------------------- Loading Routes --------------------- #
import json

with open('layer.json', 'r') as f:
    layer_json = json.load(f)

print("----- RouteLayer JSON File -------",end='\n')
print(layer_json,end='\n\n')

rl = RouteLayer.from_json("layer.json")
print(
    f"""{rl.encoder.type=}
{rl.encoder.name=}
{rl.routes=}"""
)