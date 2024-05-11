import os
import sys
import unittest

import numpy as np

# Make sure we can find the stuff without a proper python package
sys.path.append("..")


class TestSamServer(unittest.TestCase):
    def test_onnx_conversion(self):
        from bioimageio_colab.sam_server import get_sam_model, export_onnx_model

        sam = get_sam_model("vit_b")
        export_onnx_model(sam, "onnx-test.onnx", opset=12)
        assert os.path.exists("onnx-test.onnx")

    def test_compute_embeddings(self):
        from bioimageio_colab.sam_server import compute_embeddings, get_example_image

        image = get_example_image()
        embeds = compute_embeddings("vit_b", image)
        assert embeds.shape == (1, 256, 64, 64)

    def test_interactive_segmentation(self):
        from bioimageio_colab.sam_server import compute_embeddings, get_example_image, interactive_segmentation

        image = get_example_image()
        compute_embeddings("vit_b", image)

        point_coords = np.array([100, 120]).reshape((1, 1, 2))
        point_labels = np.array([1]).reshape((1, 1))
        features = interactive_segmentation("vit_b", point_coords, point_labels)
        # TODO check the features


if __name__ == "__main__":
    unittest.main()
