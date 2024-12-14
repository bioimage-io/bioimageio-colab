import os
from typing import Optional

import numpy as np
import requests
import torch
from segment_anything import SamPredictor, sam_model_registry

MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}


def load_model_from_ckpt(model_name: str, cache_dir: str) -> torch.nn.Module:
    model_url = MODELS.get(model_name, None)

    if model_url is None:
        raise ValueError(
            f"Model {model_name} not found. Available models: {list(MODELS.keys())}"
        )

    # Download model if not in cache
    basename = model_url.split("/")[-1]
    model_path = os.path.join(cache_dir, basename)

    if not os.path.exists(model_path):
        os.makedirs(cache_dir, exist_ok=True)
        response = requests.get(model_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download model from {model_url}")
        with open(model_path, "wb") as f:
            f.write(response.content)

    # Load model state
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    model_architecture = model_name[:5]
    sam = sam_model_registry[model_architecture]()
    sam.load_state_dict(ckpt)

    # Create a SAM predictor
    sam_predictor = SamPredictor(sam)
    sam_predictor.model_architecture = model_architecture

    return sam_predictor


def _to_image_format(array: np.ndarray) -> np.ndarray:
    # we require the input to be uint8
    if array.dtype != np.dtype("uint8"):
        # first normalize the input to [0, 1]
        array = array.astype("float32") - array.min()
        array = array / array.max()
        # then bring to [0, 255] and cast to uint8
        array = (array * 255).astype("uint8")
    if array.ndim == 2:
        image = np.concatenate([array[..., None]] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 3:
        image = array
    else:
        raise ValueError(
            f"Invalid input image of shape {array.shape}. Expected either 2-channel grayscale or 3-channel RGB image."
        )
    return image


def compute_embedding(
    sam_predictor: SamPredictor,
    array: np.ndarray,
) -> np.ndarray:
    # Run image encoder to compute the embedding
    image = _to_image_format(array)
    sam_predictor.set_image(image)

    return sam_predictor


def segment_image(
    sam_predictor: SamPredictor,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    # box: Optional[np.ndarray] = None,
    # mask_input: Optional[np.ndarray] = None,
):
    if isinstance(point_coords, list):
        point_coords = np.array(point_coords, dtype=np.float32)
    if isinstance(point_labels, list):
        point_labels = np.array(point_labels, dtype=np.float32)
    masks, scores, logits = sam_predictor.predict(
        point_coords=point_coords[:, ::-1],  # SAM has reversed XY conventions
        point_labels=point_labels,
        box=None,  # Not supported yet
        mask_input=None,  # Not supported yet
        multimask_output=False,
    )
    return masks


if __name__ == "__main__":
    sam_predictor = load_model_from_ckpt("vit_b_lm", "./model_cache")
    sam_predictor = compute_embedding(sam_predictor, np.random.rand(1024, 1024))
    masks = segment_image(sam_predictor, [[10, 10]], [1])
