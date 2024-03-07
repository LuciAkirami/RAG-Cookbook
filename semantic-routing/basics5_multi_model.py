'''
In this example, we will do multi-modal routing. That is, we will also include images in the embeddings
'''

from datasets import load_dataset
data = load_dataset(
    "aurelio-ai/shrek-detection", split="train", trust_remote_code=True
)

# open in notebook to view the image
# data[3]["image"]

# counting number of shrek and non shrek pics
shrek_pics = [d["image"] for d in data if d["is_shrek"]]
not_shrek_pics = [d["image"] for d in data if not d["is_shrek"]]
print(f"We have {len(shrek_pics)} shrek pics, and {len(not_shrek_pics)} not shrek pics")

from semantic_router import Route
from semantic_router.layer import RouteLayer

# defining routes
shrek = Route(
    name="shrek",
    utterances=shrek_pics,
)

not_shrek = Route(
    name="not shrek",
    utterances=not_shrek_pics,
)

routes = [shrek, not_shrek]

# instantiating the embedding, here its CLIPEncoder as it handles image embeddings
from semantic_router.encoders import CLIPEncoder
# downloads the CLIP encoder model
encoder = CLIPEncoder()

# adding routes and encoder to RouteLayer
rl = RouteLayer(routes=routes, encoder=encoder)

# testing on text
print("Testing on Text")
print("\nInput: don't you love politics?")
print(rl("don't you love politics?"))
print("\nInput: shrek")
print(rl("shrek"))
print("\nInput: dwayne the rock johnson")
print(rl("dwayne the rock johnson"),end="\n\n")

# the test dataset contains all shrek images
test_data = load_dataset(
    "aurelio-ai/shrek-detection", split="test", trust_remote_code=True
)

# testing on images
print("Testing on Images")
# viewable in notebook
# test_data[0]["image"]
print("Giving two Shrek Image:")
print(rl(test_data[0]["image"]).name)
print(rl(test_data[1]["image"]).name)