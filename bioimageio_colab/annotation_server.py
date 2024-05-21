import os
import zipfile
import requests
from imjoy_rpc.hypha import connect_to_server
import numpy as np
# Prepare paths for output
from tifffile import imread, imwrite
from kaibu_utils import features_to_mask
import urllib

hpa_data_url = "https://github.com/bioimage-io/bioimageio-colab/releases/download/v0.1/hpa-dataset-v2-98-rgb.zip"

def download_zip(url, save_path):
    """
    Download a ZIP file from the specified URL and save it to the given path.
    """
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(1024):
            file.write(data)
    print(f"Downloaded {save_path}")

def unzip_file(zip_path, extract_to):
    """
    Unzip a ZIP file to the specified directory.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted images to {extract_to}")


async def start_annotation_server(path2data: str = "./hpa_data", outpath: str = "./kaibu_annotations"):
    # Get absolute paths
    path2data = os.path.abspath(path2data)
    outpath = os.path.abspath(outpath)

    # Create the output path
    os.makedirs(outpath, exist_ok=True)

    # Check if the path2data exists
    if not os.path.exists(path2data):
        # Create the path
        os.makedirs(path2data)
        # Download the data
        save_path = os.path.join(path2data, hpa_data_url.split("/")[-1])
        download_zip(hpa_data_url, save_path)
        # Unzip the data
        unzip_file(save_path, path2data)
        # Remove the zip file
        os.remove(save_path)
        print(f"Removed {save_path}")
    
    training_images = []

    path2label = os.path.join(outpath, "labels")
    os.makedirs(path2label, exist_ok=True)

    path2source = os.path.join(outpath, "source")
    os.makedirs(path2source, exist_ok=True)
    # Connect to the server link
    server_url = "https://ai.imjoy.io"
    server = await connect_to_server({"server_url": server_url})

    # When multiple people open the link above, they can join a common workspace as an ImJoy client
    def add_image(image, label):
        training_images.append((image, label))
        print(f"{len(training_images)} available already.")
        return

    def get_random_image():
        filenames = [f for f in os.listdir(path2data) if f.endswith(".tif")]
        n = np.random.randint(len(filenames)-1)
        image = imread(os.path.join(path2data, filenames[n]))
        if len(image.shape)==3 and image.shape[0]==3:
            image = np.transpose(image, [1,2,0])
        #print(image.shape)
        new_filename = f"{len(os.listdir(path2source)) + 1}_{filenames[n]}"


        return image, filenames[n], new_filename,

    def save_annotation(filename, newname, features, image_shape):
        mask = features_to_mask(features, image_shape)
        image = imread(os.path.join(path2data, filename))
        if len(image.shape)==3 and image.shape[0]==3:
            image = np.transpose(image, [1,2,0])
        imwrite(os.path.join(path2source, newname), image)
        imwrite(os.path.join(path2label, newname), mask)


    svc = await server.register_service({
        "name": "Model Trainer",
        "id": "bioimageio-colab",
        "config": {
            "visibility": "public"
        },
        "get_random_image": get_random_image,
        "save_annotation": save_annotation,

    })
    sid = svc['id']
    config_str = f'{{"service_id": "{sid}", "server_url": "{server_url}"}}'
    encoded_config = urllib.parse.quote(config_str, safe='/', encoding=None, errors=None)
    annotator_url = 'https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/main/plugins/bioimageio-colab.imjoy.html&config=' + encoded_config
    print(annotator_url)


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    loop.create_task(start_annotation_server())

    loop.run_forever()