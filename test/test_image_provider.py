import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaibu_utils import mask_to_features

from docs.data_providing_service import get_random_image, save_annotation


def test_load_image():
    supported_file_types = ("tif", "tiff")
    image, image_name = get_random_image(
        image_folder="./bioimageio_colab/",
        supported_file_types=supported_file_types,
    )
    assert image is not None
    assert isinstance(image, np.ndarray)
    assert image.shape == (512, 512, 3)
    assert image_name is not None
    assert isinstance(image_name, str)
    assert image_name == "example_image"


# def test_save_annotation():
#     mask = np.zeros((512, 512))
#     mask[10:20, 10:20] = 1
#     features = mask_to_features(mask)

#     save_annotation(
#         annotations_folder="./",
#         image_name="test_image",
#         features=features,  # square coordinates
#         image_shape=(512, 512)
#     )
#     assert os.path.exists("test_image_mask_1.tif")
#     os.remove("test_image_mask_1.tif")
#     assert True
