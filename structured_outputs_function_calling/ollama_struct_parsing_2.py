from openai import OpenAI
import datetime
from pydantic import BaseModel, Field
from typing import List
from rich import print
import instructor

# patching the openai library to output json
client = instructor.patch(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

# ------------ Defining Your Data ----------
class Person(BaseModel):
    name: str
    age: int

class PersonBirthday(BaseModel):
    name: str
    age: int
    birthday: datetime.date

class Address(BaseModel):
    address: str = Field(description="Full street address")
    city: str
    state: str

# the address of the person extends the person
class PersonAddress(Person):
    address: Address

# iterating for 20 times to check how well its performing
for _ in range(20):
    try:
        resp = client.chat.completions.create(
            model="openhermes",
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Today is {datetime.date.today()}
    
                    Extract `Jason Liu is thirty years old his birthday is yesturday`
                    he lives at 123 Main St, San Francisco, CA""",
                    # here we can obersve that we are providing information within back ticks (`)
                    # this makes the LLM understands that we want to extract information from this info
                },
            ],
            response_model=PersonAddress,
        )

        print(resp.model_dump_json(indent=2))
    except Exception as e:
        print("Error")