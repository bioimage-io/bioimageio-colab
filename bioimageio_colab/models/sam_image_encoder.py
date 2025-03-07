from typing import Literal

import numpy as np


class SamImageEncoder:
    def __init__(
        self, model_path: str, model_architecture: Literal["vit_b", "vit_l", "vit_h"]
    ):
        import torch
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide

        # Extract image encoder from checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        build_sam = sam_model_registry[model_architecture]
        sam = build_sam(checkpoint=model_path).to(device)
        self.image_encoder = sam.image_encoder
        self.device = sam.device

        # Define image transform and normalization
        self._transform = ResizeLongestSide(sam.image_encoder.img_size)
        self._normalize_and_pad = sam.preprocess

    def _to_image_format(self, array: np.ndarray) -> np.ndarray:
        # Convert input to np.ndarray
        if not isinstance(array, np.ndarray):
            array = np.array(array)

        if array.ndim == 2:
            # Convert grayscale image to RGB
            array = np.concatenate([array[..., None]] * 3, axis=-1)
        elif array.ndim == 3:
            if array.shape[0] == 3:
                # Convert from CHW to HWC
                array = np.transpose(array, [1, 2, 0])
        else:
            raise ValueError(
                f"Invalid input image of shape {array.shape}. Expected either 2-channel (HxW) grayscale or 3-channel (HxWxC) RGB image."
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

    def _preprocess(self, array: np.array) -> tuple:
        """
        Preprocess the input image before feeding it to the model.

        Args:
            array (np.ndarray): Input image in either 2-channel grayscale (H, W) or 3-channel RGB (H, W, 3) format.
        Returns:
            tuple: A tuple containing the following elements:
                - input_image_torch (torch.Tensor): The preprocessed input image as a torch tensor (BCHW)
                - original_image_shape (tuple): The size of the original input image (H, W)
                - sam_scale (float): The scale factor used to resize the image to the correct size
        """
        import torch

        # Convert input to image format (HWC, uint8)
        original_image = self._to_image_format(array)
        original_image_shape = original_image.shape[:2]
        # Resize the image to the correct size
        resized_image = self._transform.apply_image(original_image)
        sam_scale = self._transform.target_length / max(original_image_shape)
        # Convert the image to a torch tensor
        input_image_torch = torch.as_tensor(resized_image, device=self.device)
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
        input_image_torch = self._normalize_and_pad(input_image_torch)

        return input_image_torch, original_image_shape, sam_scale

    def encode(self, array: np.ndarray) -> dict:
        """
        Encode an image using the SAM image encoder.

        Args:
            array (np.ndarray): Input image in either 2-channel grayscale (H, W) or 3-channel RGB (H, W, 3) format.

        Returns:
            dict: A dictionary containing the following keys:
                - "features" (np.ndarray): The extracted features from the image
                - "input_size" (tuple): The size of the input image (H, W)
        """
        import torch

        # Preprocess the input image
        input_image_torch, original_image_shape, sam_scale = self._preprocess(array)

        # Run inference
        with torch.no_grad():
            features = self.image_encoder(input_image_torch)

        return {
            "features": features.cpu().numpy(),
            "original_image_shape": original_image_shape,
            "sam_scale": sam_scale,
        }


if __name__ == "__main__":
    from tifffile import imread

    # Deploy the model
    model = SamImageEncoder(
        model_path="./.model_cache/sam_vit_b.pt",
        model_architecture="vit_b",
    )

    image_array = imread("./data/example_image.tif")

    result = model.encode(image_array)
    print(result.keys())
    print(result["features"].shape)
    print(result["input_size"])
