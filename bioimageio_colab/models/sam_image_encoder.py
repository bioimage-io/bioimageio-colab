from ray import serve
from typing import Literal
import torch
import os
import requests
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_gpus": 1},
)
class SAM_image_encoder:
    def __init__(
        self,
        model_path: str,
        model_url: str,
        model_architecture: Literal["vit_b", "vit_l", "vit_h"],
    ):
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

    def _load_model(self, model_path: str, model_architecture: str) -> torch.nn.Module:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_architecture](checkpoint=model_path)
        return sam.to(device)

    def __call__(self, original_image: np.ndarray):
        """image in RGB format (np.ndarray with shape (H, W, 3) and dtype uint8)"""
        # Validate image shape and dtype
        assert isinstance(original_image, np.ndarray), "Image should be a numpy array."
        assert original_image.dtype == np.uint8, "Image array should have dtype uint8."
        assert (
            original_image.shape[-1] == 3
        ), "Image image should have 3 channels (RGB)."

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


if __name__ == "__main__":
    # Comment out lines 11-16 to test this class without Ray Serve
    from tifffile import imread

    # Deploy the model
    models = {
        "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
        "sam_vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
    }
    model_name = "sam_vit_b"
    cache_dir = "./.model_cache"
    model_path = os.path.join(cache_dir, f"{model_name}.pt")

    deployment = SAM_image_encoder(
        model_path=model_path,
        model_url=models[model_name],
        model_architecture="vit_b",
    )

    image = imread("./bioimageio_colab/example_image.tif")
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])

    result = deployment(image)
    print(result)
