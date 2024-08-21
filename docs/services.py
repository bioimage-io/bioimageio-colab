import asyncio
import os
import urllib

import numpy as np
from hypha_rpc import connect_to_server
from kaibu_utils import features_to_mask
from tifffile import imread, imwrite

# Required packages: ['hypha-rpc', 'kaibu-utils==0.1.14', 'tifffile==2024.7.24']

SERVER_URL = "https://hypha.aicell.io"
PLUGIN_URL = "https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/chatbot_extension/plugins/bioimageio-colab-annotator.imjoy.html"
PATH_TO_DATA = "/mnt/"
PATH_TO_ANNOTATIONS = os.path.join(PATH_TO_DATA, "annotations")
os.makedirs(PATH_TO_ANNOTATIONS, exist_ok=True)  # Make sure the annotations folder exists


def list_image_files():
    return [f for f in os.listdir(PATH_TO_DATA) if f.endswith(".tif")]


def read_image(path):
    image = imread(os.path.join(PATH_TO_DATA, path))
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    return image


def get_random_image():
    filenames = list_image_files()
    r = np.random.randint(len(filenames) - 1)
    file_name = filenames[r]
    image = read_image(file_name)
    return (image, file_name.split(".")[0])


def save_annotation(image_name, features, image_shape):
    mask = features_to_mask(features, image_shape)
    n_image_masks = len(
        [f for f in os.listdir(PATH_TO_ANNOTATIONS) if f.startswith(image_name)]
    )
    mask_name = os.pth.join(
        PATH_TO_ANNOTATIONS, f"{image_name}_mask_{n_image_masks + 1}.tif"
    )
    imwrite(mask_name, mask)


def upload_image_to_s3():
    """
    Steps:
    - Create a user prefix on S3
    - Create a data and annotation prefix
    - For every image:
        - Load the image from the data folder into a numpy array
        - Upload the image to the data prefix

    Return:
    - The user prefix

    # TODO: register a data providing service on K8S cluster that uses the user prefix (get_random_image_s3, save_annotation_s3)
    """
    raise NotImplementedError


async def register_service(server_url, token):
    # Connect to the server link
    server = await connect_to_server({"server_url": server_url, "token": token})

    # Generate token for the current workspace
    token = await server.generate_token()

    # Register the service
    svc = await server.register_service(
        {
            "name": "Collaborative Annotation",
            "id": "data-provider",
            "config": {
                "visibility": "public",  # TODO: make protected
                "run_in_executor": True,
            },
            # Exposed functions:
            # get a random image from the dataset
            # returns the image as a numpy image
            "get_random_image": get_random_image,
            # save the annotation mask
            # pass the filename of the image, the new filename, the features and the image shape
            "save_annotation": save_annotation,
        }
    )

    # Create the annotator URL
    annotation_sid = svc["id"]
    config_str = f'{{"server_url": "{SERVER_URL}", "annotation_service_id": "{annotation_sid}", "token": "{token}"}}'
    encoded_config = urllib.parse.quote(
        config_str, safe="/", encoding=None, errors=None
    )
    annotator_url = f"https://imjoy.io/lite?plugin={PLUGIN_URL}&config={encoded_config}"

    # Option 1: Return the annotator URL to stdout
    print(annotator_url)

    # Option 2: Save the annotator URL to a file
    with open("/mnt/annotator_url.txt", "w") as f:
        f.write(annotator_url)
        # Doesn't show up in mounted folder

    # Option 3: Save the annotator URL to the local storage
    # js.localStorage.setItem("annotator_url", annotator_url)

    # Option 4: Send the annotator URL to the main thread
    # js.self.postMessage({"type": "annotator_url", "value": annotator_url})

    # Option 5: Send the annotator URL via hypha service
    # requires the registration of a service in the main thread

