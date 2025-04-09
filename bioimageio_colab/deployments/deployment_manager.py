import socket
import sys
from pathlib import Path
from typing import Dict, List, Optional
from time import sleep

import ray
from ray import serve

from bioimageio_colab.utils import create_logger, logging_format


class DeploymentManager:
    def __init__(self, address: str, serve_port: int = 8000, **init_kwargs):
        self.logger = create_logger("DeploymentManager")

        # Connect to Ray
        self._connect_to_ray(address, serve_port, **init_kwargs)

        # List all available deployments
        self._deployments = self._list_deployments()

    @property
    def deployments(self) -> List[str]:
        return list(self._deployments.keys())

    @property
    def deployment_status(self) -> Dict[str, serve.Deployment]:
        """Get the current deployment status"""
        serve_status = ray.serve.status()
        return serve_status.applications

    def _connect_to_ray(self, address: str, serve_port: int, **kwargs) -> None:
        """Connect to Ray with the specified runtime environment"""
        # Check if Ray is already initialized
        while ray.is_initialized():
            self.logger.info("Ray is already initialized. Shutting down...")
            ray.shutdown()
            sleep(1)

        # Connect to Ray
        self.logger.info("Connecting to Ray...")
        ray.init(
            address=address,
            # Add the bioimageio_colab module to the runtime env
            runtime_env={"py_modules": [str(Path(__file__).parent.parent)]},
            logging_format=logging_format,
            **kwargs,
        )

        # Log the file package URI
        runtime_env_info = ray.get_runtime_context().runtime_env
        self._package_uri = runtime_env_info.get("py_modules", [None])[0]
        if self._package_uri:
            self.logger.info(
                f"Successfully pushed 'bioimageio_colab' module to: {self._package_uri}"
            )

        python_version = sys.version.split()[0]
        self.logger.info(
            f"Successfully connected to Ray (Ray version: {ray.__version__}, Python version: {python_version})"
        )

        # Check if Ray Serve is already running, if not start it
        if not self._check_ray_serve():
            self.logger.info("Ray Serve is not initialized. Starting Serve...")
            serve.start(
                http_options={
                    "host": "0.0.0.0",
                    "port": self._find_port(serve_port, step=100),
                },
                # logging_config=logging_format,  # TODO: Fix logging
            )

    def _check_ray_serve(self) -> bool:
        """Check if Ray Serve is running"""
        try:
            serve.context._get_global_client()
            return True
        except serve.exceptions.RayServeException:
            return False

    def _find_port(self, port: int, step: int) -> int:
        """Find next available port starting from given port number.

        Args:
            port: Starting port number to check
            step: Increment between port numbers to check

        Returns:
            First available port number
        """
        available = False
        out_port = port
        while not available:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", out_port)) != 0:
                    available = True
                else:
                    out_port += step
        if out_port != port:
            self.logger.warning(
                f"Port {port} is not available. Using {out_port} instead."
            )
        return out_port

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
        deployment_options: Optional[Dict] = None,
    ) -> None:
        """Deploy an application to Ray Serve"""
        if not ray.is_initialized():
            raise RuntimeError("Not connected to Ray")

        # Get the deployment class
        try:
            deployment_class = self._deployments[deployment]
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
        deployment_class = deployment_class.options(**(deployment_options or {}))

        # Access and update ray_actor_options to include the package URI
        ray_actor_options = deployment_class.ray_actor_options or {}
        runtime_env = ray_actor_options.setdefault("runtime_env", {})
        runtime_env.setdefault("py_modules", []).append(self._package_uri)
        deployment_class = deployment_class.options(ray_actor_options=ray_actor_options)

        # Bind keyword arguments to the deployment class
        app = deployment_class.bind(**(deployment_kwargs or {}))

        # Deploy the application
        self.logger.info(f"Deploying '{deployment}'...")
        serve.run(app, name=deployment, route_prefix=None)

        # Test the deployment
        self.get_handle(deployment)

        if deployment in applications:
            self.logger.info(f"Updated application deployment '{deployment}'")
        else:
            self.logger.info(f"Deployed new application '{deployment}'")

    def undeploy(self, deployment: str) -> None:
        """Undeploy an application from Ray Serve"""
        if not ray.is_initialized():
            raise RuntimeError("Not connected to Ray")

        # Check if application exists
        serve_status = serve.status()
        applications = serve_status.applications

        if deployment in applications:
            self.logger.info(f"Deleting deployment '{deployment}'")
            serve.delete(deployment)
        else:
            raise ValueError(
                f"Deployment '{deployment}' not found in deployed applications"
            )

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
    import asyncio

    from tifffile import imread

    base_dir = Path(__file__).parent.parent.parent
    temp_dir = base_dir / "ray_tmp"
    cache_dir = base_dir / ".model_cache"

    deployment_manager = DeploymentManager(
        address=None,
        _temp_dir=str(temp_dir),
    )

    deployment_manager.deploy(
        deployment="SamInference",
        deployment_kwargs={
            "cache_dir": str(cache_dir),
        },
    )

    handle = deployment_manager.get_handle("SamInference")

    # Test the handle
    test_image_path = base_dir / "data/example_image.tif"
    image = imread(test_image_path)

    async def test_handle():
        res = await handle.options(multiplexed_model_id="sam_vit_b").remote(image)
        assert isinstance(res, dict)
        assert "features" in res

        res = await handle.segment_image.options(
            multiplexed_model_id="sam_vit_b_lm"
        ).remote(image)
        assert isinstance(res, dict)
        assert "mask" in res

        res = await handle.get_onnx_model.options(
            multiplexed_model_id="sam_vit_b_em_organelles"
        ).remote(quantize=False)
        assert isinstance(res, bytes)

    asyncio.run(test_handle())
