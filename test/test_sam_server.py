import os
import sys
import unittest

import numpy as np
from bioimageio_colab_server.sam_server import get_sam_model, export_onnx_model, compute_embeddings, get_example_image, segment
from bioimageio_colab_server.hypha_data_store import HyphaDataStore

ds = HyphaDataStore()

class TestSamServer(unittest.TestCase):
    def test_onnx_conversion(self):
        sam = get_sam_model("vit_b")
        export_onnx_model(sam, "onnx-test.onnx", opset=12)
        assert os.path.exists("onnx-test.onnx")

    def test_compute_embeddings(self):
        image = get_example_image()
        embeds = compute_embeddings(ds, "vit_b", image, return_embeddings=True)
        assert embeds.shape == (1, 256, 64, 64)

    def test_segment(self):
        image = get_example_image()
        
        predictor_id = compute_embeddings(ds, "vit_b", image, return_embeddings=False)

        point_coords = [[100, 120]]
        point_labels = [1]
        features = segment(ds, predictor_id, point_coords, point_labels)
        assert isinstance(features, list)


if __name__ == "__main__":
    unittest.main()
