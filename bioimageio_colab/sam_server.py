import os
import warnings
from functools import partial

import imageio.v3 as imageio
import numpy as np
import requests
import torch

from kaibu_utils import mask_to_features
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

from bioimageio_colab.hypha_data_store import HyphaDataStore

from logging import getLogger

logger = getLogger(__name__)
logger.setLevel("DEBUG")

IMAGE_URL = "https://owncloud.gwdg.de/index.php/s/fSaOJIOYjmFBjPM/download"
MODELS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/staged/1/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/staged/1/files/vit_b.pt",
}


def get_model_names():
    """Get the names of available SAM models.
    """
    return list(MODELS.keys())


def get_sam_model(model_name):
    """Get the model from SAM / micro_sam for the given name.
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


def export_onnx_model(
    sam,
    output_path,
    opset: int,
    return_single_mask: bool = True,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
) -> None:
    logger.info(f"Exporting model to ONNX with opset {opset}...")
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules:
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size

    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output_path, "wb") as f:
            print(f"Exporting onnx model to {output_path}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    logger.info(f"Exported model to {output_path}")


def get_example_image():
    image = imageio.imread(IMAGE_URL)
    return np.asarray(image)


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
        raise ValueError(f"Invalid input image of shape {input_.shape}. Expect either 2D grayscale or 3D RGB image.")
    return image


def compute_embeddings(ds, model_name, image=None, return_embeddings=False):
    logger.info(f"Computing embeddings for model {model_name}...")
    sam = get_sam_model(model_name)
    predictor = SamPredictor(sam)
    predictor.reset_image()
    if image is None:
        image = get_example_image()
    predictor.set_image(_to_image(image))
    if return_embeddings:
        image_embeddings = predictor.get_image_embedding().cpu().numpy()
        logger.info(f"Computed embeddings of shape {image_embeddings.shape}")
        return image_embeddings
    else:
        # Set the global state to use precomputed stuff for server side interactive segmentation.
        predictor_id = ds.put("object", predictor, 'predictor')
        logger.info(f"Caching predictor with ID {predictor_id}")
        # TODO: remove predictor
        return predictor_id


async def get_onnx(ds, model_name, opset_version=12):
    output_path = f"{model_name}.onnx"
    if not os.path.exists(output_path):
        sam = get_sam_model(model_name)
        export_onnx_model(sam, output_path, opset=opset_version)

    file_id = ds.put("file", f"file://{output_path}", output_path)
    url = ds.get_url(file_id)
    return url


def segment(ds, predictor_id, point_coordinates, point_labels):
    logger.info(f"Segmenting with predictor {predictor_id}...")
    logger.debug(f"Point coordinates: {point_coordinates}, {point_labels}")
    if isinstance(point_coordinates, list):
        point_coordinates = np.array(point_coordinates, dtype=np.float32)
    if isinstance(point_labels, list):
        point_labels = np.array(point_labels, dtype=np.float32)
    predictor = ds.get(predictor_id)['value']
    mask, scores, logits = predictor.predict(
        point_coords=point_coordinates[:, ::-1],  # SAM has reversed XY conventions
        point_labels=point_labels,
        multimask_output=False
    )
    logger.info(f"Predicted mask of shape {mask.shape}")
    features = mask_to_features(mask[0])
    return features


async def start_server():
    from imjoy_rpc.hypha import connect_to_server, login

    server_url = "https://ai.imjoy.io"

    token = None # await login({"server_url": server_url})
    server = await connect_to_server({"server_url": server_url, "token": token})

    # Upload to hypha.
    ds = HyphaDataStore()
    await ds.setup(server)

    svc = await server.register_service({
        "name": "Sam Server",
        "id": "bioimageio-colab-sam",
        "config": {
            "visibility": "public"
        },
        # Exposed functions:
        # get the names of available models
        # returns a list of names
        "get_model_names": get_model_names,
        # get an example image for interactive segmentation:
        # returns the image as a numpy image
        "get_example_image": get_example_image,
        # compute the image embeddings:
        # pass the model-name and the image to compute the embeddings on
        "compute_embeddings": partial(compute_embeddings, ds),
        # get the prompt encoder and mask decoder in onnx format
        # pass the model-name for which to get the onnx model
        "get_onnx": partial(get_onnx, ds),
        # run interactive segmentation based on prompts
        # NOTE: this is just for demonstration purposes. For fast annotation
        # and for reliable multi-user support you should use the ONNX model and run the interactive
        # segmentation client-side.
        # pass the model-name and the point coordinates and labels
        # returns the predicted mask encoded as geo json
        "segment": partial(segment, ds),
        "ping": lambda: "pong",
    })
    sid = svc["id"]
    print("The server was started with the following ID:")
    print(sid)


if __name__ == "__main__":
    import asyncio

    loop = asyncio.get_event_loop()
    loop.create_task(start_server())

    loop.run_forever()
