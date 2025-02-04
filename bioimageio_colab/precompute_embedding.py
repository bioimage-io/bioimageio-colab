import numpy as np
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

print(f"Image original shape: {result['original_image_shape']}")  # [512, 512]
print(f"SAM scale: {result['sam_scale']}")  # 2.0

# Save the features to a binary file
embedding = result["features"]
embedding.tofile(BINARY_PATH)
print(f"Saved features to {BINARY_PATH}")

print(
    f"Embedding: {embedding[0, 0, 0, :5]}..."
)  # [-0.00867976 -0.01164575 -0.01368209 -0.01407861 -0.01369949
print(f"Embedding shape: {embedding.shape}")  # (1, 256, 64, 64)

embedding_loaded = np.fromfile(BINARY_PATH, dtype="float32").reshape(embedding.shape)
assert np.array_equal(embedding, embedding_loaded)
assert np.allclose(embedding, embedding_loaded)
