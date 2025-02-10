import os
from argparse import ArgumentParser

import numpy as np
from hypha_rpc.sync import connect_to_server
from PIL import Image
from tifffile import imread


def precompute_embedding(args):
    # Resolve relative paths
    if not os.path.isabs(args.img_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.img_path = os.path.abspath(os.path.join(script_dir, args.img_path))

    # Connect to the server and get the service
    client = connect_to_server({"server_url": args.server_url})
    service = client.get_service(args.service_id, {"mode": "last"})

    # Load the image
    image = imread(args.img_path)
    if image.ndim == 2:
        image = np.concatenate([image[..., None]] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    else:
        raise ValueError(
            f"Invalid input image of shape {image.shape}. Expected either 2-channel grayscale or 3-channel RGB image."
        )
    assert image.dtype == np.dtype("uint8")
    assert image.ndim == 3
    assert image.shape[2] == 3

    # Compute the embedding
    print(f"Computing embedding for image of shape {image.shape}")
    result = service.compute_embedding(
        image=image,
        model_id=args.model_id,
    )

    print(f"Image original shape: {result['original_image_shape']}")
    # [512, 512]
    print(f"SAM scale: {result['sam_scale']}")
    # 2.0
    embedding = result["features"]
    print(f"Embedding: {embedding[0, 0, 0, :5]}...")
    # [-0.00867976 -0.01164575 -0.01368209 -0.01407861 -0.01369949
    print(f"Embedding shape: {embedding.shape}")
    # (1, 256, 64, 64)

    # Save the features to a binary file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fname = os.path.basename(args.img_path).replace(".tif", f"_{args.model_id}.bin")
    embed_fpath = os.path.join(base_dir, "data", fname)
    embedding.tofile(embed_fpath)
    print(f"Saved embedding to {embed_fpath}")

    # Test loading the embedding
    embedding_loaded = np.fromfile(embed_fpath, dtype="float32").reshape(
        embedding.shape
    )
    assert np.array_equal(embedding, embedding_loaded)
    assert np.allclose(embedding, embedding_loaded)

    # Save the image as png
    img = Image.fromarray(image)
    fname = os.path.basename(args.img_path).replace(".tif", ".png")
    img_fname = os.path.join(base_dir, "data", fname)
    img.save(img_fname)
    print(f"Saved PNG image to {img_fname}")

    # Test loading the image
    img_loaded = Image.open(img_fname)
    assert np.array_equal(image, np.array(img_loaded))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        default="../data/example_image.tif",
        help="Path to the input image file (absolute or relative to script location)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="sam_vit_b_lm",
        help="Model ID to use for embedding computation",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="https://hypha.aicell.io",
        help="URL of the Hypha server",
    )
    parser.add_argument(
        "--service_id",
        type=str,
        default="bioimageio-colab/microsam",
        help="Service ID for the model service",
    )
    args = parser.parse_args()
    precompute_embedding(args)
