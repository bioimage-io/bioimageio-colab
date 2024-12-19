import os
from functools import partial
from typing import Literal

import numpy as np
import requests
import torch
from ray import serve
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1},
)
class SamImageEncoder:
    def __init__(
        self,
        cache_dir: str,
        model_name: str,
        model_url: str,
        model_architecture: Literal["vit_b", "vit_l", "vit_h"],
    ):
        model_path = os.path.join(cache_dir, f"{model_name}.pt")

        # Download model if not available
        if not os.path.exists(model_path):
            self._download_model(
                model_path=model_path,
                model_url=model_url,
            )

        # Extract image encoder from checkpoint
        sam = self._load_model(model_path, model_architecture)
        self.image_encoder = sam.image_encoder
        self.image_encoder
        self.device = sam.device

        # Define image transform and preprocess
        self.transform = ResizeLongestSide(sam.image_encoder.img_size)
        self.preprocess = sam.preprocess

    def _download_model(self, model_path: str, model_url: str) -> None:
        cache_dir = os.path.dirname(model_path)
        os.makedirs(cache_dir, exist_ok=True)
        response = requests.get(model_url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download model from {model_url}")
        with open(model_path, "wb") as f:
            f.write(response.content)

    def _to_image_format(self, array: np.ndarray) -> np.ndarray:
        # Convert input to np.ndarray
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        # Convert grayscale image to RGB
        if array.ndim == 2:
            # Convert grayscale image to RGB
            array = np.concatenate([array[..., None]] * 3, axis=-1)
        elif array.ndim == 3:
            if array.shape[0] == 3:
                # Convert from CHW to HWC
                array = np.transpose(array, [1, 2, 0])
        else:
            raise ValueError(
                f"Invalid input image of shape {array.shape}. Expected either 2-channel grayscale or 3-channel RGB image."
            )

        assert (
            array.shape[-1] == 3
        ), f"Image image should have 3 channels (RGB). Got shape {array.shape}."

        # Convert input to uint8 if not already
        if array.dtype != np.dtype("uint8"):
            # first normalize the input to [0, 1] per channel
            array = array.astype("float32") - array.min(axis=(0, 1))
            array = array / array.max(axis=(0, 1))
            # then bring to [0, 255] and cast to uint8
            array = (array * 255).astype("uint8")

        return array

    def _load_model(self, model_path: str, model_architecture: str) -> torch.nn.Module:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_architecture](checkpoint=model_path)
        return sam.to(device)

    def __call__(self, array: np.ndarray):
        """image in RGB format (np.ndarray with shape (H, W, 3) and dtype uint8)"""
        # Validate image shape and dtype
        original_image = self._to_image_format(array)

        input_image = self.transform.apply_image(original_image)  # input: np.array

        input_image_torch = torch.as_tensor(input_image, device=self.device)

        # Ensure the input tensor is in the correct shape (BCHW)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]

        assert (
            len(input_image_torch.shape) == 4
            and input_image_torch.shape[1] == 3
            and max(*input_image_torch.shape[2:]) == self.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.image_encoder.img_size}."

        # Preprocess the image before feeding it to the model
        input_image_torch = self.preprocess(input_image_torch)
        input_size = tuple(input_image_torch.shape[-2:])

        # Run inference
        with torch.no_grad():
            features = self.image_encoder(input_image_torch)

        return {"features": features.cpu().numpy(), "input_size": input_size}


model_urls = {
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "sam_vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    "sam_vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
}

sam_app_registry = {
    model_name: partial(
        SamImageEncoder.options(name=model_name).bind,
        model_name=model_name,
        model_url=model_urls[model_name],
        model_architecture=model_architecture,
    )
    for model_name, model_architecture in [
        ("sam_vit_b", "vit_b"),
        ("sam_vit_b_lm", "vit_b"),
        ("sam_vit_b_em_organelles", "vit_b"),
    ]
}


if __name__ == "__main__":
    #! Comment out lines 13-17 and 128-140 to test this class without Ray Serve
    from tifffile import imread

    # Deploy the model
    model_name = "sam_vit_b"
    cache_dir = "./.model_cache"

    deployment = SamImageEncoder(
        cache_dir=cache_dir,
        model_name=model_name,
        model_url=model_urls[model_name],
        model_architecture="vit_b",
    )

    image = imread("./bioimageio_colab/example_image.tif")

    result = deployment(image)
    print(result)
