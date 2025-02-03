from hypha_rpc.sync import connect_to_server
from tifffile import imread

SERVER_URL = "https://hypha.aicell.io"
SID = "bioimageio-colab/microsam"
MODEL_ID = "sam_vit_b_lm"
IMG_PATH = "./data/example_image.tif"
BINARY_PATH = f"./data/example_image_embedding.bin"


# Connect to the server and get the service
client = connect_to_server({"server_url": SERVER_URL})
service = client.get_service(SID, {"mode": "last"})

# Load the image and compute the embedding
image = imread(IMG_PATH)
result = service.compute_embedding(
    image=image,
    model_id=MODEL_ID,
)

# Save the features to a binary file
features = result["features"]
features.tofile(BINARY_PATH)
print(f"Saved features to {BINARY_PATH}")

print(f"Features: {features[0, 0, 0, :5]}...")
print(f"Features shape: {features.shape}")
print(f"Image original shape: {result['original_image_shape']}")
print(f"SAM scale: {result['sam_scale']}")
