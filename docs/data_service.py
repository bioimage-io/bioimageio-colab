import asyncio
import io
import time
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List

import httpx
import numpy as np
from hypha_rpc import connect_to_server
from hypha_rpc.rpc import ObjectProxy
from kaibu_utils import features_to_mask
from PIL import Image
from tifffile import imread

WORKSPACE = "bioimage-io"
COLLECTION_ID = "bioimage-io/colab-annotations"


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    TIF = "tif"
    TIFF = "tiff"


def list_image_files(image_folder: Path) -> List[Path]:
    extensions = tuple(member.value for member in ImageFormat)
    return [f for f in image_folder.iterdir() if f.suffix.lower().lstrip(".") in extensions]


def _read_tiff(file_path: str) -> np.ndarray:
    return imread(file_path)


def _read_pil(file_path: str) -> np.ndarray:
    with Image.open(file_path) as img:
        return np.array(img)


_IMAGE_READERS = {
    ImageFormat.TIFF: _read_tiff,
    ImageFormat.TIF: _read_tiff,
    ImageFormat.PNG: _read_pil,
    ImageFormat.JPEG: _read_pil,
    ImageFormat.JPG: _read_pil,
}


def process_image(image: np.ndarray) -> np.ndarray:
    # Check axes
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
        image = np.transpose(image, [1, 2, 0])

    # Convert to RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]

    # Normalize to uint8
    if image.dtype != np.uint8:
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    return image


def read_image(file_path: Path) -> np.ndarray:
    ext = file_path.suffix.lower().lstrip(".")
    try:
        fmt = ImageFormat(ext)
    except ValueError:
        raise ValueError(f"Unsupported file extension: {ext}")

    reader = _IMAGE_READERS.get(fmt)
    if reader is None:
        raise NotImplementedError(f"No reader implemented for format: {fmt}")

    image = reader(str(file_path))
    return process_image(image)


async def get_image(
    server_url: str,
    artifact_manager: ObjectProxy,
    artifact_id: str,
    images_path: Path
) -> str:
    filenames = list_image_files(images_path)
    r = np.random.randint(max(len(filenames) - 1, 1))
    image_path = filenames[r]
    image = read_image(image_path)

    pil_image = Image.fromarray(image)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    image_name = image_path.stem + ".png"

    try:
        existing_images = await artifact_manager.list_files(artifact_id, dir_path="images")
        image_exists = any(f["name"] == image_name for f in existing_images)
    except Exception:
        image_exists = False

    if not image_exists:
        upload_url = await artifact_manager.put_file(artifact_id, file_path=f"images/{image_name}")

        async with httpx.AsyncClient() as client:
            await client.put(upload_url, content=img_bytes)

    artifact_alias = artifact_id.split("/")[-1]
    image_url = f"{server_url}/{WORKSPACE}/artifacts/{artifact_alias}/files/images/{image_name}"

    return image_url


async def save_annotation(
    artifact_manager: ObjectProxy,
    artifact_id: str,
    image_name: str,
    features: list,
    image_shape: tuple,
) -> None:
    mask = features_to_mask(features, image_shape[:2])

    try:
        files = await artifact_manager.list_files(artifact_id, dir_path="annotations")
        existing_masks = [f for f in files if f["name"].startswith(image_name)]
        n_image_masks = len(existing_masks)
    except Exception:
        n_image_masks = 0

    mask_filename = f"{image_name}_mask_{n_image_masks + 1}.png"
    upload_path = f"annotations/{mask_filename}"

    pil_mask = Image.fromarray(mask.astype(np.uint8))
    mask_byte_arr = io.BytesIO()
    pil_mask.save(mask_byte_arr, format="PNG")
    mask_bytes = mask_byte_arr.getvalue()

    upload_url = await artifact_manager.put_file(artifact_id, file_path=upload_path)

    async with httpx.AsyncClient() as client:
        await client.put(upload_url, content=mask_bytes)


async def register_service(
    server_url: str,
    token: str,
    name: str,
    description: str,
    images_path: str = "/mnt",
):
    # Check if the images folder exists
    images_path = Path(images_path)
    if not images_path.is_dir():
        raise FileNotFoundError("Mounted images folder not found")

    # Connect to the server
    client = await connect_to_server({"server_url": server_url, "token": token})
    artifact_manager = await client.get_service("public/artifact-manager")

    # Create artifact
    parent_id = COLLECTION_ID
    manifest = {
        "name": f"Annotation Session {name}",
        "description": description,
        "owner": {
            "id": client.config.user["id"],
            "email": client.config.user["email"],
        }
    }

    artifact = await artifact_manager.create(
        parent_id=parent_id,
        manifest=manifest,
        type="dataset",
        stage=True,
    )
    artifact_id = artifact.id
    print(f"Data Artifact ID: {artifact_id}")

    # Register the service
    svc = await client.register_service(
        {
            "name": name,
            "description": description,
            "id": "data-provider-" + str(int(time.time() * 100)),
            "type": "annotation-data-provider",
            "config": {
                "visibility": "public",
            },
            # Exposed functions:
            "get_image": partial(
                get_image,
                server_url=server_url,
                artifact_manager=artifact_manager,
                artifact_id=artifact_id,
                images_path=images_path,
            ),
            "save_annotation": partial(
                save_annotation, artifact_manager, artifact_id
            ),
        }
    )
    print(f"Data Provider Service ID: {svc['id']}")

    # TODO: Commit the artifact when the service is stopped


class PretrainedCellposeModel(str, Enum):
    CYTO = "cyto"
    CYTO3 = "cyto3"
    NUCLEI = "nuclei"
    TISSUENET_CP3 = "tissuenet_cp3"
    LIVECELL_CP3 = "livecell_cp3"
    YEAST_PHC_CP3 = "yeast_PhC_cp3"
    YEAST_BF_CP3 = "yeast_BF_cp3"
    BACT_PHASE_CP3 = "bact_phase_cp3"
    BACT_FLUOR_CP3 = "bact_fluor_cp3"
    DEEPBACS_CP3 = "deepbacs_cp3"


async def create_annotations_json(
    artifact_manager: ObjectProxy,
    artifact_id: str,
) -> str:
    import json

    # List all images
    try:
        image_files = await artifact_manager.list_files(artifact_id, dir_path="images")
        image_names = {f["name"] for f in image_files}
    except Exception:
        image_names = set()

    # List all annotations
    try:
        annotation_files = await artifact_manager.list_files(
            artifact_id, dir_path="annotations"
        )
    except Exception:
        annotation_files = []

    uploaded_images = []
    uploaded_annotations = []

    for ann in annotation_files:
        ann_name = ann["name"]
        # Assuming format: {image_stem}_mask_{n}.png
        # We need to extract image_stem.
        # Split by "_mask_" from the right
        parts = ann_name.rpartition("_mask_")
        if parts[1] != "_mask_":
            continue

        image_stem = parts[0]
        target_image_name = None

        # Check if the stem corresponds to an image
        if image_stem in image_names:
            target_image_name = image_stem
        elif f"{image_stem}.png" in image_names:
            target_image_name = f"{image_stem}.png"

        if target_image_name:
            uploaded_images.append(f"images/{target_image_name}")
            uploaded_annotations.append(f"annotations/{ann_name}")

    data = {
        "uploaded_images": uploaded_images,
        "uploaded_annotations": uploaded_annotations,
    }

    json_str = json.dumps(data, indent=2)

    # Upload annotations.json
    file_path = "annotations.json"
    upload_url = await artifact_manager.put_file(artifact_id, file_path=file_path)

    async with httpx.AsyncClient() as client:
        await client.put(upload_url, content=json_str.encode("utf-8"))


async def finetune_cellpose(
    server_url: str, token: str, data_artifact_id: str, base_model: str = "cyto3"
):

    async with connect_to_server({"server_url": server_url, "token": token}) as client:

        # Create/update annotations.json
        artifact_manager = await client.get_service("public/artifact-manager")
        await create_annotations_json(artifact_manager, data_artifact_id)

        # Access the finetuning service
        service_id = "bioimage-io/cellpose-finetuning"
        svc = await client.get_service(service_id)

        # Start the finetuning process
        status = await svc.start_training(
            artifact=data_artifact_id,
            metadata_path="annotations.json",
            model=base_model,
            ratio=0.8,
            n_samples=None,  # use all samples
            n_epochs=10,
            learning_rate=1e-6,
            weight_decay=1e-4,
        )
        if status["status_type"] != "running":
            print(f"Failed to start training: {status}")
            return

        session_id = status["session_id"]
        while True:
            status = await svc.get_training_status(session_id)
            print(f"Training status: {status['status']}")

            if status["status_type"] != "running":
                break

            await asyncio.sleep(10)  # wait for 10 seconds before checking again

        print(f"Training completed with status: {status['status_type']}")

        return session_id


if __name__ == "__main__":
    # Example: Create a data artifact from local images and annotations

    import os

    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get the workspace token from environment
    token = os.getenv("WORKSPACE_TOKEN")
    if not token:
        raise ValueError("WORKSPACE_TOKEN not found in environment variables")

    # Configuration
    server_url = "https://hypha.aicell.io"

    # Get the base directory (parent of docs/)
    base_dir = Path(__file__).parent.parent.resolve()
    data_dir = base_dir / "data" / "test_images" / "images"

    # Use the same directory for both images and masks
    # The function will find image-mask pairs based on naming pattern
    images_path = data_dir  # Contains: 29164_1526_E8_1.tif, 5625_1842_B3_65.tif

    # Example 1: Register the service
    print(f"Images path: {images_path}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    loop.run_until_complete(
        register_service(
            server_url=server_url,
            token=token,
            name="My Annotation Service",
            description="A service for annotating images",
            images_path=str(images_path),
        )
    )
    
    # Keep the loop running to serve requests
    print("Service is running. Press Ctrl+C to stop.")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Stopping service...")
    finally:
        loop.close()
