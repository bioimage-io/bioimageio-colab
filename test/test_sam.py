import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bioimageio_colab.sam import compute_embedding, load_model_from_ckpt, segment_image
from docs.data_providing_service import get_random_image


def test_sam():
    sam_predictor = load_model_from_ckpt(model_name="vit_b", cache_dir="./model_cache/")
    assert os.path.exists("./model_cache/sam_vit_b_01ec64.pth")
    image, _ = get_random_image(
        image_folder="./bioimageio_colab/",
        supported_file_types=("tif"),
    )
    sam_predictor = compute_embedding(sam_predictor, image)
    assert sam_predictor.is_image_set is True

    masks = segment_image(sam_predictor, point_coords=[[80, 80]], point_labels=[1])
    assert all([mask.shape == image.shape[:2] for mask in masks])
