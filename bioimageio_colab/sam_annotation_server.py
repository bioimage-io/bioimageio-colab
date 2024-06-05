import os
import urllib
import zipfile
from functools import partial
from logging import getLogger

import numpy as np
import requests
import torch
from imjoy_rpc.hypha import connect_to_server
from kaibu_utils import features_to_mask, mask_to_features
from segment_anything import SamPredictor, sam_model_registry
from tifffile import imread, imwrite

from bioimageio_colab.hypha_data_store import HyphaDataStore

logger = getLogger(__name__)
logger.setLevel("DEBUG")

HPA_DATA_URL = "https://github.com/bioimage-io/bioimageio-colab/releases/download/v0.1/hpa-dataset-v2-98-rgb.zip"
PATH2DATA = os.path.abspath("./hpa_data")
PATH2SOURCE = os.path.abspath("./kaibu_annotations/source")
PATH2LABEL = os.path.abspath("./kaibu_annotations/labels")
MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}


# Functions for the SAM model


def get_sam_model(model_name):
    """
    Get the model from SAM / micro_sam for the given name.
    """
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
    logger.info(f"Predicted mask of shape {mask.shape}")
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
    print(f"Downloaded {save_path}")


def unzip_file(zip_path, extract_to):
    """
    Unzip a ZIP file to the specified directory.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted images to {extract_to}")


def get_random_image():
    filenames = [f for f in os.listdir(PATH2DATA) if f.endswith(".tif")]
    n = np.random.randint(len(filenames) - 1)
    image = imread(os.path.join(PATH2DATA, filenames[n]))
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    new_filename = f"{len(os.listdir(PATH2SOURCE)) + 1}_{filenames[n]}"
    return (
        image,
        filenames[n],
        new_filename,
    )


def save_annotation(filename, newname, features, image_shape):
    mask = features_to_mask(features, image_shape)
    image = imread(os.path.join(PATH2DATA, filename))
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    imwrite(os.path.join(PATH2SOURCE, newname), image)
    imwrite(os.path.join(PATH2LABEL, newname), mask)


async def start_sam_annotation_server():
    """
    Start the SAM annotation server.

    When multiple people open the link, they can join a common workspace as an ImJoy client
    """
    # Check if the data is available
    if not os.path.exists(PATH2DATA):
        # Create the path
        os.makedirs(PATH2DATA)
        # Download the data
        save_path = os.path.join(PATH2DATA, HPA_DATA_URL.split("/")[-1])
        download_zip(HPA_DATA_URL, save_path)
        # Unzip the data
        unzip_file(save_path, PATH2DATA)
        # Remove the zip file
        os.remove(save_path)
        print(f"Removed {save_path}")

    # Create the output paths
    os.makedirs(PATH2SOURCE, exist_ok=True)
    os.makedirs(PATH2LABEL, exist_ok=True)

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
            "config": {"visibility": "public"},
            # Exposed functions:
            # get a random image from the dataset
            # returns the image as a numpy image
            "get_random_image": get_random_image,
            # save the annotation mask
            # pass the filename of the image, the new filename, the features and the image shape
            "save_annotation": save_annotation,
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
        }
    )
    sid = svc["id"]
    config_str = f'{{"service_id": "{sid}", "server_url": "{server_url}"}}'
    encoded_config = urllib.parse.quote(
        config_str, safe="/", encoding=None, errors=None
    )
    annotator_url = (
        "https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/annotation-server/plugins/bioimageio-colab-sam-annotation.imjoy.html&config="
        + encoded_config
    )
    print(annotator_url)


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    loop.create_task(start_sam_annotation_server())

    loop.run_forever()
