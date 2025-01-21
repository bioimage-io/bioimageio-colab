import numpy as np
from ray import serve

from .sam_image_encoder import SamImageEncoder

SAM_MODELS = {
    "sam_vit_b": {
        "architecture": "vit_b",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    "sam_vit_b_lm": {
        "architecture": "vit_b",
        "url": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
    },
    "sam_vit_b_em_organelles": {
        "architecture": "vit_b",
        "url": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
    },
}


# Default deployment options can be overridden by `deployment.options()`
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=1,
)
class SamDeployment:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.models = SAM_MODELS

    async def _download_model(self, model_path: str, model_url: str) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to download model from {model_url}")
                content = await response.read()
                with open(model_path, "wb") as f:
                    f.write(content)

    @serve.multiplexed(max_num_models_per_replica=2)
    async def get_model(self, model_id: str):
        import os

        model_path = os.path.join(self.cache_dir, f"{model_id}.pt")

        if not os.path.exists(model_path):
            os.makedirs(self.cache_dir, exist_ok=True)
            await self._download_model(
                model_path=model_path,
                model_url=self.models[model_id]["url"],
            )

        return SamImageEncoder(
            model_path=model_path,
            model_architecture=self.models[model_id]["architecture"],
        )

    async def __call__(self, model_id: str, array: np.ndarray):
        model = await self.get_model(model_id)
        return model.encode(array)
