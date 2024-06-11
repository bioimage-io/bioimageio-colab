import os
import urllib
import zipfile
from functools import partial
from logging import getLogger

import numpy as np
import requests
import shutil
import torch
from imjoy_rpc.hypha import connect_to_server
from kaibu_utils import features_to_mask, mask_to_features
from tifffile import imread, imwrite

from bioimageio_colab.hypha_data_store import HyphaDataStore

logger = getLogger(__name__)
logger.setLevel("INFO")

HPA_DATA_URL = "https://github.com/bioimage-io/bioimageio-colab/releases/download/v0.1/hpa-dataset-v2-98-rgb.zip"
MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}


def download_labels(ds, data_folder):
    # Zip the output folder
    zip_filename = f"{data_folder}.zip"
    shutil.make_archive(data_folder, 'zip', data_folder)
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


# Functions for the SAM model


def get_sam_model(model_name):
    """
    Get the model from SAM / micro_sam for the given name.
    """
    from segment_anything import sam_model_registry

    model_url = MODELS[model_name]
    checkpoint_path = f"{model_name}.pt"

    if not os.path.exists(checkpoint_path):
        logger.info(f"Downloading model from {model_url} to {checkpoint_path}...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    logger.info(f"Loading model {model_name} from {checkpoint_path}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = model_name[:5]
    sam = sam_model_registry[model_type]()
    ckpt = torch.load(checkpoint_path, map_location=device)
    sam.load_state_dict(ckpt)
    logger.info(f"Loaded model {model_name} from {checkpoint_path}")
    return sam


def _to_image(input_):
    # we require the input to be uint8
    if input_.dtype != np.dtype("uint8"):
        # first normalize the input to [0, 1]
        input_ = input_.astype("float32") - input_.min()
        input_ = input_ / input_.max()
        # then bring to [0, 255] and cast to uint8
        input_ = (input_ * 255).astype("uint8")
    if input_.ndim == 2:
        image = np.concatenate([input_[..., None]] * 3, axis=-1)
    elif input_.ndim == 3 and input_.shape[-1] == 3:
        image = input_
    else:
        raise ValueError(
            f"Invalid input image of shape {input_.shape}. Expect either 2D grayscale or 3D RGB image."
        )
    return image


def load_model(ds, model_name):
    sam = get_sam_model(model_name)
    sam_id = ds.put("object", sam, "sam")
    logger.info(f"Caching SAM with ID {sam_id}")
    return sam_id


def compute_embeddings(ds, sam_id, image):
    from segment_anything import SamPredictor
    
    logger.info(f"Computing embeddings for model {sam_id}...")
    sam = ds.get(sam_id)["value"]
    predictor = SamPredictor(sam)
    predictor.reset_image()
    predictor.set_image(_to_image(image))
    # Set the global state to use precomputed stuff for server side interactive segmentation.
    predictor_id = ds.put("object", predictor, "predictor")
    logger.info(f"Caching predictor with ID {predictor_id}")
    # TODO: remove predictor
    return predictor_id


def segment(ds, predictor_id, point_coordinates, point_labels):
    logger.info(f"Segmenting with predictor {predictor_id}...")
    logger.debug(f"Point coordinates: {point_coordinates}, {point_labels}")
    if isinstance(point_coordinates, list):
        point_coordinates = np.array(point_coordinates, dtype=np.float32)
    if isinstance(point_labels, list):
        point_labels = np.array(point_labels, dtype=np.float32)
    predictor = ds.get(predictor_id)["value"]
    mask, scores, logits = predictor.predict(
        point_coords=point_coordinates[:, ::-1],  # SAM has reversed XY conventions
        point_labels=point_labels,
        multimask_output=False,
    )
    logger.debug(f"Predicted mask of shape {mask.shape}")
    features = mask_to_features(mask[0])
    return features


# Functions for collaborative annotation
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


def save_annotation(path2data, path2source, path2label, filename, newname, features, image_shape):
    mask = features_to_mask(features, image_shape)
    image = imread(os.path.join(path2data, filename))
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    imwrite(os.path.join(path2source, newname), image)
    imwrite(os.path.join(path2label, newname), mask)


async def start_server(data_url: str, path2data: str="./data", outpath:str="./kaibu_annotations"):
    """
    Start the SAM annotation server.

    When multiple people open the link, they can join a common workspace as an ImJoy client
    """
    # Check if the data is available
    path2data = os.path.abspath(path2data)
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

    svc = await server.register_service(
        {
            "name": "Model Trainer",
            "id": "bioimageio-colab",
            "config": {"visibility": "public", "run_in_executor": True},
            # Exposed functions:
            # get a random image from the dataset
            # returns the image as a numpy image
            "get_random_image": partial(get_random_image, path2data, path2source),
            # save the annotation mask
            # pass the filename of the image, the new filename, the features and the image shape
            "save_annotation": partial(save_annotation, path2data, path2source, path2label),
            # load the model
            # pass the model-name to load the model
            # returns the model-id
            "load_model": partial(load_model, ds),
            # compute the image embeddings:
            # pass the model-id and the image to compute the embeddings on
            "compute_embeddings": partial(compute_embeddings, ds),
            # run interactive segmentation based on prompts
            # NOTE: this is just for demonstration purposes. For fast annotation
            # and for reliable multi-user support you should use the ONNX model and run the interactive
            # segmentation client-side.
            # pass the predictor-id the point coordinates and labels
            # returns the predicted mask encoded as geo json
            "segment": partial(segment, ds),
            # download the source and label images as a zip file
            "download_labels": partial(download_labels, ds, outpath),
        }
    )
    sid = svc["id"]
    config_str = f'{{"service_id": "{sid}", "server_url": "{server_url}"}}'
    encoded_config = urllib.parse.quote(
        config_str, safe="/", encoding=None, errors=None
    )
    annotator_url = (
        "https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/main/plugins/bioimageio-colab.imjoy.html&config="
        + encoded_config
    )
    print("-" * 80)
    print("Annotation server is running at:")
    print(annotator_url)
    print("-" * 80)
    print("To download the annotated labels, go to:")
    print(f"{server_url}/{server.config['workspace']}/services/{sid.split(':')[1]}/download_labels")
    print("-" * 80)

if __name__ == "__main__":
    import asyncio
    logger.setLevel("DEBUG")

    loop = asyncio.get_event_loop()
    loop.create_task(start_server(data_url=HPA_DATA_URL))

    loop.run_forever()
