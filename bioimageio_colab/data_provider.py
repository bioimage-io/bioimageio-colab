import os
import urllib
import zipfile
from functools import partial
from logging import getLogger

import numpy as np
import requests
import shutil
from imjoy_rpc.hypha import connect_to_server
from kaibu_utils import features_to_mask
from tifffile import imread, imwrite

from bioimageio_colab.hypha_data_store import HyphaDataStore

logger = getLogger(__name__)
logger.setLevel("INFO")

HPA_DATA_URL = "https://github.com/bioimage-io/bioimageio-colab/releases/download/v0.1/hpa-dataset-v2-98-rgb.zip"


def download_zip(url, save_path):
    """
    Download a ZIP file from the specified URL and save it to the given path.
    """
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(1024):
            file.write(data)

    logger.info(f"Downloaded {save_path}")


def unzip_file(zip_path, extract_to):
    """
    Unzip a ZIP file to the specified directory.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info(f"Extracted images to {extract_to}")


def get_random_image(path2data, path2source):
    filenames = [f for f in os.listdir(path2data) if f.endswith(".tif")]
    n = np.random.randint(len(filenames) - 1)
    image = imread(os.path.join(path2data, filenames[n]))
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    new_filename = f"{len(os.listdir(path2source)) + 1}_{filenames[n]}"
    return (
        image,
        filenames[n],
        new_filename,
    )


def save_annotation(
    path2data, path2source, path2label, filename, newname, features, image_shape
):
    mask = features_to_mask(features, image_shape)
    image = imread(os.path.join(path2data, filename))
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    imwrite(os.path.join(path2source, newname), image)
    imwrite(os.path.join(path2label, newname), mask)


def download_labels(ds, data_folder):
    # Zip the output folder
    zip_filename = f"{data_folder}.zip"
    shutil.make_archive(data_folder, "zip", data_folder)
    logger.info(f"Zipped {data_folder} to {zip_filename}")

    # Upload the zip file to HyphaDataStore
    with open(zip_filename, "rb") as f:
        zip_file_content = f.read()
    zip_file_id = ds.put("file", zip_file_content, zip_filename)
    logger.info(f"Uploaded zip file with ID {zip_file_id}")

    # Generate the URL for the zip file
    zip_file_url = ds.get_url(zip_file_id)
    logger.info(f"Download URL for zip file: {zip_file_url}")
    return zip_file_url


async def start_server(
    path2data: str = "./data",
    outpath: str = "./kaibu_annotations",
    data_url: str = None,
):
    """
    Start the SAM annotation server.

    When multiple people open the link, they can join a common workspace as an ImJoy client
    """
    path2data = os.path.abspath(path2data)
    if data_url is not None:
        # Check if the data is available
        if not os.path.exists(path2data):
            # Create the path
            os.makedirs(path2data)
            # Download the data
            save_path = os.path.join(path2data, data_url.split("/")[-1])
            download_zip(data_url, save_path)
            # Unzip the data
            unzip_file(save_path, path2data)
            # Remove the zip file
            os.remove(save_path)
            logger.info(f"Removed {save_path}")
        else:
            logger.info(f"Data already exists at {path2data}")

    # Create the output paths
    path2source = os.path.abspath(os.path.join(outpath, "source"))
    path2label = os.path.abspath(os.path.join(outpath, "labels"))
    os.makedirs(path2source, exist_ok=True)
    os.makedirs(path2label, exist_ok=True)

    # Connect to the server link
    server_url = "https://ai.imjoy.io"
    server = await connect_to_server({"server_url": server_url})

    # Upload to hypha.
    ds = HyphaDataStore()
    await ds.setup(server)

    # Generate token for the current workspace
    token = await server.generate_token()

    # Register the service
    svc = await server.register_service(
        {
            "name": "Collaborative Annotation",
            "id": "bioimageio-colab-annotation",
            "config": {
                "visibility": "public",  # TODO: make protected
                "run_in_executor": True,
            },
            # Exposed functions:
            # get a random image from the dataset
            # returns the image as a numpy image
            "get_random_image": partial(get_random_image, path2data, path2source),
            # save the annotation mask
            # pass the filename of the image, the new filename, the features and the image shape
            "save_annotation": partial(
                save_annotation, path2data, path2source, path2label
            ),
            # download the source and label images as a zip file
            "download_labels": partial(download_labels, ds, outpath),
        }
    )
    annotation_sid = svc["id"]
    config_str = f'{{"server_url": "{server_url}", "annotation_service_id": "{annotation_sid}", "token": "{token}"}}'
    encoded_config = urllib.parse.quote(
        config_str, safe="/", encoding=None, errors=None
    )
    annotator_url = (
        "https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/chatbot_extension/plugins/bioimageio-colab-annotator.imjoy.html&config="
        + encoded_config
    )
    print("-" * 80)
    print("Annotation server is running at:")
    print(annotator_url)
    print("-" * 80)
    print("To download the annotated labels, go to:")
    print(
        f"{server_url}/{server.config['workspace']}/services/{annotation_sid.split(':')[1]}/download_labels"
    )
    print("-" * 80)


if __name__ == "__main__":
    import asyncio

    logger.setLevel("DEBUG")

    loop = asyncio.get_event_loop()
    loop.create_task(start_server(data_url=HPA_DATA_URL))

    loop.run_forever()
