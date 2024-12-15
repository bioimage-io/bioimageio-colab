import imageio.v3 as imageio
import numpy as np
from micro_sam.util import get_sam_model, precompute_image_embeddings

img = imageio.imread("./example_image.png")
model = get_sam_model(model_type="vit_b_lm")
embed = precompute_image_embeddings(model, img, ndim=2)["features"]

fname = "emebddings_vit_b_lm.bin"
embed.tofile(fname)
embed_loaded = np.fromfile(fname, dtype="float32").reshape(embed.shape)

assert np.allclose(embed, embed_loaded)
