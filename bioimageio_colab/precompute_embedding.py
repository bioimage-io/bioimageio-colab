from hypha_rpc.sync import connect_to_server
from tifffile import imread
import msgpack


SERVER_URL = "https://hypha.aicell.io"
SID = "bioimageio-colab/microsam"
MODEL_ID = "sam_vit_b_lm"
IMG_PATH = "./data/example_image.tif"
BINARY_PATH = f"./data/example_image_embedding_{MODEL_ID}.bin"


# Connect to the server and get the service
client = connect_to_server({"server_url": SERVER_URL})
service = client.get_service(SID, {"mode": "first"})


# Load the image and compute the embedding
image = imread(IMG_PATH)
result = service.compute_embedding(
    image=image,
    model_id=MODEL_ID,
)

# Add the model_id to the result and encode it
result["model_id"] = MODEL_ID
b_object = client.rpc._encode(result)

# Save the result to binary file using msgpack
with open(BINARY_PATH, "wb") as f:
    packed_data = msgpack.packb(b_object, use_bin_type=True)
    f.write(packed_data)
