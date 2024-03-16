import outlines
from outlines import models

model = models.llamacpp(
    "/home/topisano/Desktop/projects/models/openhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=-1,  # to use GPU acceleration
    seed=1337,  # to set a specific seed
)

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just sucks!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
print(answer)

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is beautiful!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
print(answer)