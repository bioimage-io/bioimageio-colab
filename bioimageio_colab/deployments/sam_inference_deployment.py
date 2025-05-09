import os
from pathlib import Path
from typing import Optional

import numpy as np
from ray import serve

from bioimageio_colab.deployments.sam_inference_model import SamInferenceModel
from bioimageio_colab.deployments.utils import parse_requirements

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
    ray_actor_options={
        "num_gpus": 0 if os.getenv("DISABLE_GPU", "false") == "true" else 1,
        "num_cpus": 1,
        # "memory": None,
        # Deployment specific requirements
        "runtime_env": {
            "pip": parse_requirements(
                Path(__file__).parent.parent.parent / "requirements-sam.txt"
            ),
        },
    },
    max_ongoing_requests=1,
    max_queued_requests=10,
)
class SamInference:
    def __init__(self, cache_dir: str = "/tmp/ray/.model_cache"):
        self.cache_dir = Path(cache_dir)
        self.models = SAM_MODELS
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def _download_model(self, model_path: Path, model_url: str) -> None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(model_url) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to download model from {model_url}")
                content = await response.read()
                model_path.write_bytes(content)

    @serve.multiplexed(max_num_models_per_replica=1)
    async def _get_model(self, model_id: str):
        model_path = self.cache_dir / f"{model_id}.pt"

        if not model_path.exists():
            await self._download_model(
                model_path=model_path,
                model_url=self.models[model_id]["url"],
            )

        return SamInferenceModel(
            model_path=str(model_path),
            model_architecture=self.models[model_id]["architecture"],
        )

    async def __call__(self, array: np.ndarray) -> dict:
        return await self.encode_image(array)

    async def encode_image(self, array: np.ndarray) -> dict:
        model_id = serve.get_multiplexed_model_id()
        model = await self._get_model(model_id)
        return model.encode_image(array)

    async def get_onnx_model(
        self,
        quantize: bool = True,
    ) -> bytes:
        model_id = serve.get_multiplexed_model_id()
        model = await self._get_model(model_id)
        return model.get_onnx_model(
            quantize=quantize,
            gelu_approximate=False,
        )

    async def segment_image(
        self,
        array: np.ndarray,
        points_per_side: Optional[int] = 32,
        pred_iou_thresh: Optional[float] = 0.88,
        stability_score_thresh: Optional[float] = 0.95,
        min_mask_region_area: Optional[int] = 0,
    ) -> dict:
        model_id = serve.get_multiplexed_model_id()
        model = await self._get_model(model_id)
        return model.segment_image(
            array,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
