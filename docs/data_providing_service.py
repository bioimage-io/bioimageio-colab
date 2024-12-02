import os
import json
from typing import Tuple
import time
from functools import partial

import numpy as np
from hypha_rpc import connect_to_server
from kaibu_utils import features_to_mask
from tifffile import imread, imwrite


def list_image_files(image_folder: str, supported_file_types: Tuple[str]):
    return [f for f in os.listdir(image_folder) if f.endswith(supported_file_types)]


def read_image(file_path: str):
    image = imread(file_path)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    return image


def get_random_image(image_folder: str, supported_file_types: Tuple[str]):
    filenames = list_image_files(image_folder, supported_file_types)
    r = np.random.randint(max(len(filenames) - 1, 1))
    file_name = filenames[r]
    image = read_image(os.path.join(image_folder, file_name))
    return (image, file_name.split(".")[0])


def save_annotation(annotations_folder: str, image_name: str, features, image_shape):
    mask = features_to_mask(features, image_shape)
    n_image_masks = len(
        [f for f in os.listdir(annotations_folder) if f.startswith(image_name)]
    )
    mask_name = os.path.join(
        annotations_folder, f"{image_name}_mask_{n_image_masks + 1}.tif"
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

async def register_service(
    server_url: str,
    token: str,
    supported_file_types_json: str,
    name: str,
    description: str,
):
    # Define path to images and annotations
    images_path = "/mnt"
    annotations_path = "/mnt/annotations"

    # Check if the images folder exists
    if not os.path.isdir(images_path):
        raise FileNotFoundError("Mounted images folder not found")

    # Decode the JSON string to a Python tuple
    supported_file_types = tuple(json.loads(supported_file_types_json))

    # Connect to the server link
    server = await connect_to_server({"server_url": server_url, "token": token})

    # Register the service
    svc = await server.register_service(
        {
            "name": name,
            "description": description,
            "id": "data-provider-" + str(int(time.time()*100)),
            "type": "annotation-data-provider",
            "config": {
                "visibility": "public",  # TODO: make protected
                "run_in_executor": True,
            },
            # Exposed functions:
            # get a random image from the dataset
            # returns the image as a numpy image
            "get_random_image": partial(
                get_random_image, images_path, supported_file_types
            ),
            # save the annotation mask
            # pass the filename of the image, the new filename, the features and the image shape
            "save_annotation": partial(save_annotation, annotations_path),
        }
    )
    print(f"Service registered with ID: {svc['id']}")