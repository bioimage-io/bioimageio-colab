import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np


class SamInferenceModel:
    def __init__(
        self,
        model_path: str,
        model_architecture: Literal["vit_b", "vit_l", "vit_h"],
    ):
        """Initialize SAM model and all its components for efficient inference."""

        import torch
        from segment_anything import sam_model_registry

        # Load SAM model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_architecture](checkpoint=model_path)
        self.sam.to(self.device)
        self.model_path = model_path

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

    def encode_image(self, array: np.ndarray) -> dict:
        """Encode an image using the SAM image encoder."""
        from segment_anything import SamPredictor

        try:
            # Convert input to image format
            image = self._to_image_format(array)

            # Get original image shape and SAM scale
            original_image_shape = image.shape[:2]
            target_length = self.sam.image_encoder.img_size
            sam_scale = target_length / max(original_image_shape)

            # Create new predictor and encode image
            predictor = SamPredictor(self.sam)
            predictor.set_image(image)
            features = predictor.get_image_embedding()

            output = {
                "features": features.cpu().numpy(),
                "original_image_shape": original_image_shape,
                "sam_scale": sam_scale,
                "mask_threshold": self.sam.mask_threshold,
            }
        except Exception as e:
            raise e
        finally:
            predictor.reset_image()

        return output

    def get_onnx_model(
        self,
        quantize: bool = True,
        gelu_approximate: bool = False,
    ) -> bytes:
        """Export SAM model to ONNX format or load existing ONNX model.

        Args:
            quantize: Whether to quantize the model for better runtime performance
            return_single_mask: If True, only return the best mask
            gelu_approximate: Use GELU approximation for better runtime compatibility

        Returns:
            bytes: The ONNX model as bytes
        """
        import torch
        from onnxruntime import InferenceSession
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic
        from segment_anything.utils.onnx import SamOnnxModel

        onnx_path = Path(self.model_path).with_suffix(".onnx")

        if not onnx_path.exists():
            # Temporarily move model to CPU for ONNX export
            self.sam.to("cpu")

            try:
                # Create ONNX model
                onnx_model = SamOnnxModel(
                    model=self.sam,
                    return_single_mask=True,
                    use_stability_score=False,
                    return_extra_metrics=False,
                )

                # Apply GELU approximation if requested
                if gelu_approximate:
                    for n, m in onnx_model.named_modules():
                        if isinstance(m, torch.nn.GELU):
                            m.approximate = "tanh"

                # Prepare dummy inputs
                embed_dim = self.sam.prompt_encoder.embed_dim
                embed_size = self.sam.prompt_encoder.image_embedding_size
                mask_input_size = [4 * x for x in embed_size]
                dummy_inputs = {
                    "image_embeddings": torch.randn(
                        1, embed_dim, *embed_size, dtype=torch.float
                    ),
                    "point_coords": torch.randint(
                        low=0, high=1024, size=(1, 5, 2), dtype=torch.float
                    ),
                    "point_labels": torch.randint(
                        low=0, high=4, size=(1, 5), dtype=torch.float
                    ),
                    "mask_input": torch.randn(
                        1, 1, *mask_input_size, dtype=torch.float
                    ),
                    "has_mask_input": torch.tensor([1], dtype=torch.float),
                    "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float),
                }

                dynamic_axes = {
                    "point_coords": {1: "num_points"},
                    "point_labels": {1: "num_points"},
                }

                # Export ONNX model
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_path,
                        export_params=True,
                        verbose=False,
                        opset_version=17,  # Ensure opset version is >= 11
                        do_constant_folding=True,
                        input_names=list(dummy_inputs.keys()),
                        output_names=["masks", "iou_predictions", "low_res_masks"],
                        dynamic_axes=dynamic_axes,
                    )

                # Validate the model with onnxruntime
                ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
                providers = ["CPUExecutionProvider"]
                ort_session = InferenceSession(str(onnx_path), providers=providers)
                _ = ort_session.run(None, ort_inputs)

                # Quantize if requested
                if quantize:
                    quantized_path = onnx_path.with_stem(onnx_path.stem + "_quantized")
                    quantize_dynamic(
                        model_input=str(onnx_path),
                        model_output=str(quantized_path),
                        # optimize_model=True,
                        per_channel=False,
                        reduce_range=False,
                        weight_type=QuantType.QUInt8,
                    )
                    onnx_path = quantized_path

            finally:
                # Restore original device
                self.sam.to(self.device)

        # Read and return the final model
        return onnx_path.read_bytes()

    def segment_image(
        self,
        array: np.ndarray,
        points_per_side: Optional[int] = 32,
        pred_iou_thresh: Optional[float] = 0.88,
        stability_score_thresh: Optional[float] = 0.95,
        min_mask_region_area: Optional[int] = 0,
    ) -> dict:
        """
        Perform automatic segmentation on the image.
        Optional parameters override the defaults.
        """
        from segment_anything import SamAutomaticMaskGenerator

        try:
            # Convert input to image format
            image = self._to_image_format(array)

            # Create new mask generator with parameters
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
                output_mode="binary_mask",
            )

            # Generate masks
            annotations = mask_generator.generate(image)

            # Convert annotations to binary masks
            mask = np.zeros_like(image[..., 0], dtype=np.uint16)
            for mask_id, annotation in enumerate(annotations):
                mask[annotation["segmentation"]] = mask_id + 1

        except Exception as e:
            raise e

        return {
            "mask": mask,
            "annotations": annotations,
        }


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    from onnxruntime import InferenceSession

    # Test paths
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / ".model_cache/sam_vit_b_lm.pt"
    test_image_path = base_dir / "data/example_image.tif"

    # Initialize model
    model = SamInferenceModel(
        model_path=str(model_path),
        model_architecture="vit_b",
    )

    # Test image encoding
    from tifffile import imread

    image = imread(test_image_path)

    # Convert image to proper RGB format
    image_rgb = model._to_image_format(image)

    # Measure encoding time
    start_time = time.time()
    result = model.encode_image(image_rgb)
    encode_time = time.time() - start_time

    print(f"Encoding time: {encode_time:.2f}s")
    print("Result keys:", result.keys())
    print("Features shape:", result["features"].shape)
    print("Original image shape:", result["original_image_shape"])
    print("SAM scale:", result["sam_scale"])

    # Test ONNX model export
    onnx_model_bytes = model.get_onnx_model(quantize=True)
    print("\nONNX model exported successfully.")

    # Test load the ONNX model from bytes
    ort_model = InferenceSession(onnx_model_bytes, providers=["CPUExecutionProvider"])

    input_point = np.array([[80, 80]], dtype=np.float32)
    input_label = np.array([1], dtype=np.float32)

    # Convert input to the format expected by the ONNX model
    onnx_coord = np.concatenate(
        [input_point * result["sam_scale"], np.array([[0.0, 0.0]], dtype=np.float32)],
        axis=0,
    )[None, :, :].astype(np.float32)

    onnx_label = np.concatenate(
        [input_label, np.array([-1], dtype=np.float32)], axis=0
    )[None, :].astype(np.float32)

    # Prepare inputs for the ONNX model - ensure all inputs are float32
    ort_inputs = {
        "image_embeddings": result["features"].astype(np.float32),
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
        "orig_im_size": np.array(result["original_image_shape"], dtype=np.float32),
    }

    # Run inference with the ONNX model
    mask, iou_predictions, low_res_masks = ort_model.run(None, ort_inputs)

    # Convert mask to binary format
    mask = mask[0].squeeze() > result["mask_threshold"]

    # Plot image, mask, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Generated Mask")
    axes[1].axis("off")
    axes[2].imshow(image_rgb)
    axes[2].imshow(mask, alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
    print("Figure displayed.")
    plt.close(fig)

    # Test segmentation
    start_time = time.time()
    result = model.segment_image(image_rgb)
    segment_time = time.time() - start_time

    print(f"\nSegmentation time: {segment_time:.2f}s")
    print("Result keys:", result.keys())
    print("Number of masks generated:", len(result["annotations"]))

    # Plot image, mask, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(result["mask"], cmap="gray")
    axes[1].set_title("Generated Mask")
    axes[1].axis("off")
    axes[2].imshow(image_rgb)
    axes[2].imshow(result["mask"], alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
    print("Figure displayed.")
    plt.close(fig)
