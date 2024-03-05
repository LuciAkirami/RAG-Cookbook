from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import OpenAIEncoder
from dotenv import load_dotenv
from rich import print, print_json

# loading openai api key
load_dotenv()

# let's define two static routes
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

# initiating the encoder/embedder
encoder = OpenAIEncoder()

# defining RouteLayer and adding the static routes
rl = RouteLayer(
    encoder=encoder,
    routes=[education, technology]
)


# let's define a function for dynamic route
from datetime import datetime
from zoneinfo import ZoneInfo

def get_time(timezone: str) -> str:
    """Returns the current time in the given timezone

    :param timezone: The timezone to find the current time in, should
    be a valid timezone from IANA Time Zone Database like "America/New_York"
    or "Europe/London" or "Asia/HongKong" or "Australia/Sydney" or "Japan/Tokyo".
    Do NOT put the place name itself like "rome" or "new york", you must provide
    the IANA format.

    :type timezone: str
    :return: The current time in the given timezone."""

    current_time = datetime.now(ZoneInfo(timezone))
    return current_time.strftime("%H:%M")

# getting the schema of the dynamic route
from semantic_router.utils.function_call import get_schema

schema = get_schema(get_time)
print("Schema:\n")
print_json(data=schema)
print("\n")

# creating the Route for dynamic route
time_route = Route(
    name="get_time",
    utterances=[
        "what is the time in new york city?",
        "what is the time in london?",
        "I live in Rome, what time is it?",
    ],
    function_schema=schema,
)

# adding the dynamic route to the RouteLayer
rl.add(time_route)

# testing
response = rl("what is the time in new york city?")
time = get_time(**response.function_call)
print("Query: what is the time in new york city?")
print("Response: ", response.function_call)
print("Time: ", time,end="\n\n")

response = rl("what is the time in london?")
time = get_time(**response.function_call)
print("Query: what is the time in london?")
print("Response: ", response.function_call)
print("Time: ", time,end="\n\n")

response = rl("What are the lastest advancements in the Artificial Intelligence")
print("Query: What are the lastest advancements in the Artificial Intelligence")
print("Response: ", response)
print(f"Route: {response.name}",end="\n\n")

response = rl("I live in Chennai, what time is it?")
time = get_time(**response.function_call)
print("Query: I live in Chennai, what time is it?")
print("Response: ", response.function_call)
print("Time: ", time,end="\n\n")