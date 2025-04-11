import asyncio
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import ray
from dotenv import find_dotenv, load_dotenv
from hypha_rpc import connect_to_server

from bioimageio_colab.deployments.deployment_manager import DeploymentManager
from bioimageio_colab.utils import create_logger, format_time

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


class SAMService:
    def __init__(self, args: dict) -> None:
        self.logger = create_logger("SAMService")
        self.args = args
        self.deployment_manager = DeploymentManager(
            address=args.ray_address, **eval(args.ray_init_kwargs)
        )

        # Initialize service attributes
        self.deployment_name = None
        self.registration_time_s = None
        self.service_id = None

    def deploy_to_ray(self, deployment_name: str = "SamInference") -> None:
        self.deployment_name = deployment_name
        deployment_kwargs = {"cache_dir": os.path.abspath(self.args.cache_dir)}
        deployment_options = {
            "num_replicas": self.args.num_replicas,
            "max_replicas_per_node": 1,
            "max_queued_requests": self.args.max_concurrent_requests,
        }
        self.deployment_manager.deploy(
            deployment=self.deployment_name,
            restart=self.args.restart_deployment,
            deployment_kwargs=deployment_kwargs,
            deployment_options=deployment_options,
        )

    async def deployment_status(
        self,
        assert_status: bool = False,
        context: dict = None,
    ) -> dict:
        try:
            output = {}

            application = self.deployment_manager.deployment_status[
                self.deployment_name
            ]
            formatted_time = format_time(application.last_deployed_time_s)
            output[f"application: {self.deployment_name}"] = {
                "status": application.status.value,
                "last_deployed_at": formatted_time["last_deployed_at"],
                "duration_since": formatted_time["duration_since"],
            }

            deployments = application.deployments
            for name, deployment in deployments.items():
                output[f"application: {self.deployment_name}"][
                    f"deployment: {name}"
                ] = {
                    "status": deployment.status.value,
                    "replica_states": deployment.replica_states,
                }

            formatted_time = format_time(self.registration_time_s)
            output["hypha_service"] = {
                "status": "RUNNING",
                "service_id": self.service_id,
                "last_registered_at": formatted_time["last_deployed_at"],
                "duration_since": formatted_time["duration_since"],
            }

            if assert_status:
                assert application.status == "RUNNING"
                for deployment in deployments.values():
                    assert deployment.status == "HEALTHY"
                    assert deployment.replica_states["RUNNING"] > 0

            return output
        except Exception as e:
            self.logger.error(f"Error checking deployment status: {e}")
            raise e

    async def compute_image_embedding(
        self,
        semaphore: asyncio.Semaphore,
        image: np.ndarray,
        model_id: str,
        context: dict = None,
    ) -> dict:
        try:
            user = context["user"]
            if self.args.require_login and user["is_anonymous"]:
                raise PermissionError("You must be logged in to use this service.")
            user_id = user["id"]

            self.logger.info(
                f"User '{user_id}' - Putting image into the object store..."
            )
            obj_ref = ray.put(image)
            del image

            async with semaphore:
                self.logger.info(
                    f"User '{user_id}' - Computing embedding (model: '{model_id}')..."
                )
                handle = self.deployment_manager.get_handle(self.deployment_name)
                result = await handle.encode_image.options(
                    multiplexed_model_id=model_id
                ).remote(obj_ref)
                self.logger.info(f"User '{user_id}' - Embedding computed successfully.")
                return result
        except Exception as e:
            self.logger.error(f"User '{user_id}' - Error computing embedding: {e}")
            raise e

    async def get_onnx_model(
        self,
        semaphore: asyncio.Semaphore,
        model_id: str,
        quantize: bool = True,
        context: dict = None,
    ) -> bytes:
        """
        Get the ONNX model for the given model ID.
        """
        try:
            user = context["user"]
            if self.args.require_login and user["is_anonymous"]:
                raise PermissionError("You must be logged in to use this service.")
            user_id = user["id"]

            self.logger.info(
                f"User '{user_id}' - Fetching ONNX model (model: '{model_id}')..."
            )
            async with semaphore:
                handle = self.deployment_manager.get_handle(self.deployment_name)
                result = await handle.get_onnx_model.options(
                    multiplexed_model_id=model_id
                ).remote(quantize=quantize)
                self.logger.info(f"User '{user_id}' - ONNX model fetched successfully.")
                return result
        except Exception as e:
            self.logger.error(f"User '{user_id}' - Error fetching ONNX model: {e}")
            raise e

    async def segment_image(
        self,
        semaphore: asyncio.Semaphore,
        image: np.ndarray,
        model_id: str,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 0,
        context: dict = None,
    ) -> dict:
        """
        Segment an image using the given model ID.
        """
        try:
            user = context["user"]
            if self.args.require_login and user["is_anonymous"]:
                raise PermissionError("You must be logged in to use this service.")
            user_id = user["id"]

            self.logger.info(
                f"User '{user_id}' - Putting image into the object store..."
            )
            obj_ref = ray.put(image)
            del image

            async with semaphore:
                self.logger.info(
                    f"User '{user_id}' - Segmenting image (model: '{model_id}')..."
                )
                handle = self.deployment_manager.get_handle(self.deployment_name)
                result = await handle.segment_image.options(
                    multiplexed_model_id=model_id
                ).remote(
                    obj_ref,
                    points_per_side=points_per_side,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    min_mask_region_area=min_mask_region_area,
                )
                self.logger.info(f"User '{user_id}' - Image segmented successfully.")
                return result
        except Exception as e:
            self.logger.error(f"User '{user_id}' - Error segmenting image: {e}")
            raise e

    async def register_service(self) -> None:
        if self.deployment_name is None:
            raise RuntimeError(
                "Deployment not initialized. Please call deploy_to_ray() first."
            )

        self.logger.info("Registering the SAM annotation service...")
        workspace_token = self.args.token or os.environ.get("WORKSPACE_TOKEN")
        if not workspace_token:
            raise ValueError(
                "Workspace token is required to connect to the Hypha server."
            )

        client = await connect_to_server(
            {
                "server_url": self.args.server_url,
                "workspace": self.args.workspace_name,
                "name": "SAM Server",
                "token": workspace_token,
                "ping_interval": None,
            }
        )
        client_id = client.config["client_id"]
        workspace = client.config["workspace"]
        client_base_url = f"{self.args.server_url}/{workspace}/services/{client_id}"
        self.logger.info(
            f"Connected to workspace '{workspace}' with client ID: {client_id}"
        )

        semaphore = asyncio.Semaphore(self.args.max_concurrent_requests)
        self.logger.info(
            f"Created semaphore for {self.args.max_concurrent_requests} concurrent requests."
        )

        self.logger.info(
            f"Registering the SAM service: ID='{self.args.service_id}', require_login={self.args.require_login}"
        )
        service_info = await client.register_service(
            {
                "name": "Interactive Segmentation",
                "id": self.args.service_id,
                "config": {
                    "visibility": "public",
                    "require_context": True,
                    "run_in_executor": False,
                },
                "hello": lambda context=None: "Welcome to the Interactive Segmentation service!",
                "ping": lambda context=None: "pong",
                "deployment_status": self.deployment_status,
                "compute_embedding": partial(
                    self.compute_image_embedding, semaphore=semaphore
                ),
                "get_onnx_model": partial(self.get_onnx_model, semaphore=semaphore),
                "segment_image": partial(self.segment_image, semaphore=semaphore),
            }
        )

        self.service_id = service_info["id"]
        self.logger.info(f"Service registered with ID: {self.service_id}")
        self.logger.info(
            f"Test the service here: {client_base_url}:{self.args.service_id}/hello"
        )
        self.logger.info(
            f"Check deployment status: {client_base_url}:{self.args.service_id}/deployment_status"
        )
        self.registration_time_s = datetime.now(timezone.utc).timestamp()

        # Write service ID and timestamp to file
        service_file = Path(__file__).parent.parent / "service_id.txt"
        service_file.write_text(self.service_id)

        await client.serve()
