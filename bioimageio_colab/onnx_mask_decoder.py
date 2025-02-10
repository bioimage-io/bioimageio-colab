# Extra pip installations:
# pip install opencv-python
# pip install onnxruntime

# %% Imports
import os
import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from hypha_rpc.sync import connect_to_server
from kaibu_utils import mask_to_features
from tifffile import imread


# %% Functions
def load_image(file_path):
    # Resolve relative paths
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(script_dir, file_path))

    # Load the image
    image = imread(file_path)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    return image


def load_sam_decoder(
    model_id: str = "sam_vit_b_lm",
    output_dir: str = "../data",
):
    model_urls = {
        "sam_vit_b_lm": "https://raw.githubusercontent.com/constantinpape/mbexc-review/refs/heads/master/vit_b_lm_decoder.onnx",
        "sam_vit_b_em_organelles": "https://raw.githubusercontent.com/constantinpape/mbexc-review/refs/heads/master/vit_b_em_decoder.onnx",
    }
    model_url = model_urls[model_id]
    file_name = os.path.basename(model_url)
    local_model_path = os.path.join(output_dir, file_name)

    if not os.path.isabs(local_model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.abspath(os.path.join(script_dir, local_model_path))

    if not os.path.exists(local_model_path):
        urllib.request.urlretrieve(model_url, local_model_path)

    model = ort.InferenceSession(local_model_path)

    return model


def prepare_model_data(embedding_result, coordinates):
    # Check the coordinates
    if len(coordinates) != 2:
        raise ValueError(
            f"Invalid coordinates. Expected a 2-element array, but received {len(coordinates)}"
        )
    if not isinstance(coordinates[0], (int, float)) or coordinates[0] < 0:
        raise ValueError(
            f"Invalid x-coordinate. Expected a non-negative number, but received {coordinates[0]}"
        )
    if not isinstance(coordinates[1], (int, float)) or coordinates[1] < 0:
        raise ValueError(
            f"Invalid y-coordinate. Expected a non-negative number, but received {coordinates[1]}"
        )

    # Create click points
    clicks = [{"x": coordinates[0], "y": coordinates[1], "clickType": 1}]

    # Check there are input click prompts
    n = len(clicks)

    # Initialize arrays for points and labels
    point_coords = np.zeros((2 * (n + 1),), dtype=np.float32)
    point_labels = np.zeros((n + 1,), dtype=np.float32)

    # Add clicks and scale coordinates
    for i in range(n):
        point_coords[2 * i] = clicks[i]["x"] * embedding_result["sam_scale"]
        point_coords[2 * i + 1] = clicks[i]["y"] * embedding_result["sam_scale"]
        point_labels[i] = clicks[i]["clickType"]

    # Add the padding point
    point_coords[2 * n] = 0.0
    point_coords[2 * n + 1] = 0.0
    point_labels[n] = -1.0

    # Reshape point coordinates and create tensors
    point_coords = point_coords.reshape(1, n + 1, 2)
    point_labels = point_labels.reshape(1, n + 1)

    return {
        "image_embeddings": embedding_result["features"],
        "point_coords": point_coords,
        "point_labels": point_labels,
        "orig_im_size": np.array(
            embedding_result["original_image_shape"], dtype=np.float32
        ),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
    }


# %%
model_id = "sam_vit_b_lm"

image = load_image("../data/example_image.tif")

client = connect_to_server({"server_url": "https://hypha.aicell.io"})
svc = client.get_service("bioimageio-colab/microsam", {"mode": "last"})

embedding_result = svc.compute_embedding(
    image=image,
    model_id=model_id,
)
embedding = embedding_result["features"]
assert embedding.shape == (1, 256, 64, 64)
assert embedding.dtype == np.float32

model = load_sam_decoder(model_id)

coordinates = (80, 80)
feeds = prepare_model_data(embedding_result, coordinates)

output_names = ["masks"]
masks = model.run(output_names, feeds)

# %% Plot the mask
mask = masks[0].squeeze()

# Display the mask using matplotlib
plt.imshow(mask, cmap="viridis")
plt.colorbar(label="Normalized Intensity")
plt.title("Visualized Mask")
plt.axis("off")
plt.show()

# %%

# Convert the mask to a binary mask
binary_mask = (mask > 0).astype(np.uint8)

# Find contours
contours, hierarchy = cv2.findContours(
    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Create an empty image to draw contours
contour_image = np.zeros_like(binary_mask)

# Draw contours on the empty image
cv2.drawContours(
    contour_image, contours, -1, (255), thickness=1
)  # Thickness can be adjusted

# Visualize the contours
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Binary Mask")
plt.imshow(binary_mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Contours")
plt.imshow(contour_image, cmap="gray")
plt.axis("off")
plt.show()

# %%
# Reverse the color channels to match matplotlib's default color mapping
image_overlay = image[..., ::-1].copy()

# Draw contours on the RGB image
cv2.drawContours(
    image_overlay, contours, -1, (255, 0, 255), thickness=2
)

# Plot the overlay
plt.figure(figsize=(8, 8))
plt.title("Overlay of Contours on Original Image")
plt.imshow(
    cv2.cvtColor(image_overlay, cv2.COLOR_BGR2RGB)
)  # Convert BGR to RGB for plotting
plt.axis("off")
plt.show()


# %%
features = mask_to_features(binary_mask)
# %%
