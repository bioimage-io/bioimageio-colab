from typing import Optional, Dict
import sys

import ray
from ray import serve

from bioimageio_colab.utils import create_logger, logging_format


class DeploymentManager:
    def __init__(self, address: str, serve_port: int = 8000, **init_kwargs):
        self.logger = create_logger("DeploymentManager")

        # Connect to Ray
        self._connect_to_ray(address, serve_port, **init_kwargs)

        # List all available deployments
        self.deployments = self._list_deployments()

    def _connect_to_ray(self, address: str, serve_port: int, **kwargs) -> None:
        """Connect to Ray with the specified runtime environment"""
        self.logger.info("Connecting to Ray...")

        # Check if Ray is already initialized
        if ray.is_initialized():
            self.logger.info("Ray is already initialized. Shutting down...")
            ray.shutdown()

        # Connect to Ray
        ray.init(address=address, logging_format=logging_format, **kwargs)

        python_version = sys.version.split()[0]
        self.logger.info(
            f"Successfully connected to Ray (Ray version: {ray.__version__}, Python version: {python_version})"
        )

        # Stop Ray Serve if it was initialized
        try:
            serve.context._get_global_client()
        except serve.exceptions.RayServeException:
            self.logger.info("Ray Serve is not initialized. Starting Serve...")
            serve.start(
                http_options={
                    "host": "0.0.0.0",
                    "port": serve_port,
                },
                # logging_config=logging_format,  # TODO: Fix logging
            )

    def _list_deployments(self) -> Dict[str, serve.Deployment]:
        """List all deployments in the current directory that end with deployment.py"""

        deployments = {}
        current_dir = Path(__file__).parent

        # Find all files ending with deployment.py
        deployment_files = current_dir.glob("*deployment.py")

        for file in deployment_files:
            # Skip __init__.py files
            if file.name.startswith("__"):
                continue

            try:
                # Import the deployment module dynamically
                import importlib.util

                spec = importlib.util.spec_from_file_location("deployment", str(file))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for serve.Deployment object in the module
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if isinstance(item, serve.Deployment):
                        deployments[item_name] = item
                        self.logger.info(f"Loaded deployment '{item_name}' from {file}")
                        break

            except Exception as e:
                self.logger.warning(f"Failed to load deployment from {file}: {str(e)}")

        return deployments

    def deploy(
        self,
        deployment: str,
        restart: bool = False,
        deployment_kwargs: Optional[Dict] = None,
        deployment_config: Optional[Dict] = None,
    ) -> None:
        """Deploy an application to Ray Serve"""
        if not ray.is_initialized():
            raise RuntimeError("Not connected to Ray")

        # Get the deployment class
        try:
            deployment_class = self.deployments[deployment]
        except KeyError:
            raise ValueError(
                f"Deployment '{deployment}' not found in the loaded deployments"
            )

        # Check if application exists
        serve_status = serve.status()
        applications = serve_status.applications

        if deployment in applications:
            if restart:
                self.logger.info(f"Deleting existing deployment '{deployment}'")
                serve.delete(deployment)
            else:
                self.logger.info(
                    f"Deployment '{deployment}' already exists. Skipping..."
                )
                return

        # Configure deployment options
        app = deployment_class.options(**(deployment_config or {})).bind(
            **(deployment_kwargs or {})
        )

        self.logger.info(f"Deploying '{deployment}'...")

        # Deploy the application
        serve.run(app, name=deployment, route_prefix=None)

        # Test the deployment
        self.get_handle(deployment)

        if deployment in applications:
            self.logger.info(f"Updated application deployment '{deployment}'")
        else:
            self.logger.info(f"Deployed new application '{deployment}'")

    def get_handle(self, deployment: str) -> serve.handle.DeploymentHandle:
        """Get handle for an existing deployment"""
        if not ray.is_initialized():
            raise RuntimeError("Not connected to Ray. Call connect_to_ray() first.")

        # Get handle
        try:
            deployment_handle = serve.get_app_handle(name=deployment)
            assert deployment_handle is not None
        except Exception as e:
            self.logger.error(
                f"Failed to get handle for deployment '{deployment}': {str(e)}"
            )
            raise e

        return deployment_handle


if __name__ == "__main__":
    from pathlib import Path
    import asyncio
    from tifffile import imread
    import numpy as np

    base_dir = Path(__file__).parent.parent.parent
    temp_dir = base_dir / "ray_tmp"
    cache_dir = base_dir / ".model_cache"

    deployment_manager = DeploymentManager(
        address=None,
        serve_port=8100,
        _temp_dir=str(temp_dir),
    )

    deployment_manager.deploy(
        deployment="SamInferenceDeployment",
        deployment_kwargs={
            "cache_dir": str(cache_dir),
        },
        deployment_config={
            "num_replicas": 1,
            "max_replicas_per_node": 1,
            "max_queued_requests": 10,
        },
    )

    handle = deployment_manager.get_handle("SamInferenceDeployment")

    # Test the handle
    test_image_path = base_dir / "data/example_image.tif"
    image = imread(test_image_path)

    async def test_handle():
        res = await handle.options(multiplexed_model_id="sam_vit_b").remote(image)
        assert isinstance(res, dict)
        assert "features" in res

        res = await handle.segment_image.options(multiplexed_model_id="sam_vit_b_lm").remote(image)
        assert isinstance(res, dict)
        assert "mask" in res

        res = await handle.get_onnx_model.options(multiplexed_model_id="sam_vit_b_em_organelles").remote(quantize=False)
        assert isinstance(res, bytes)

    asyncio.run(test_handle())
