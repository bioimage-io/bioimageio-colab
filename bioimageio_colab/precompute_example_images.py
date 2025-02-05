import os

import numpy as np
from hypha_rpc.sync import connect_to_server
from PIL import Image
from tifffile import imread

MODEL_ID = "sam_vit_b_lm"
IMG_PATH = "./data/example_image.tif"


# Connect to the server and get the service
server_url = "https://hypha.aicell.io"
client = connect_to_server({"server_url": server_url})
sid = "bioimageio-colab/microsam"
service = client.get_service(sid, {"mode": "last"})

# Load the image
image = imread(IMG_PATH)
if image.ndim == 2:
    image = np.concatenate([image[..., None]] * 3, axis=-1)
elif image.ndim == 3 and image.shape[0] == 3:
    image = np.transpose(image, [1, 2, 0])
else:
    raise ValueError(
        f"Invalid input image of shape {image.shape}. Expected either 2-channel grayscale or 3-channel RGB image."
    )
assert image.dtype == np.dtype("uint8")
assert image.ndim == 3
assert image.shape[2] == 3

# Compute the embedding
print(f"Computing embedding for image of shape {image.shape}")
result = service.compute_embedding(
    image=image,
    model_id=MODEL_ID,
)

print(f"Image original shape: {result['original_image_shape']}")
# [512, 512]
print(f"SAM scale: {result['sam_scale']}")
# 2.0
embedding = result["features"]
print(f"Embedding: {embedding[0, 0, 0, :5]}...")
# [-0.00867976 -0.01164575 -0.01368209 -0.01407861 -0.01369949
print(f"Embedding shape: {embedding.shape}")
# (1, 256, 64, 64)

# Save the features to a binary file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fname = os.path.basename(IMG_PATH).replace(".tif", f"_{MODEL_ID}.bin")
embed_fpath = os.path.join(base_dir, "data", fname)
embedding.tofile(embed_fpath)
print(f"Saved embedding to {embed_fpath}")

# Test loading the embedding
embedding_loaded = np.fromfile(embed_fpath, dtype="float32").reshape(embedding.shape)
assert np.array_equal(embedding, embedding_loaded)
assert np.allclose(embedding, embedding_loaded)

# Save the image as png
img = Image.fromarray(image)
fname = os.path.basename(IMG_PATH).replace(".tif", ".png")
img_fname = os.path.join(base_dir, "data", fname)
img.save(img_fname)
print(f"Saved PNG image to {img_fname}")

# Test loading the image
img_loaded = Image.open(img_fname)
assert np.array_equal(image, np.array(img_loaded))
