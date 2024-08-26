import asyncio
import os

import requests
from hypha_rpc import connect_to_server
from tifffile import imread

BASE_URL = "https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/"
BRANCH = "main"

# Download the register_sam_service.py file
script_url = os.path.join(BASE_URL, BRANCH, "bioimageio_colab/register_sam_service.py")
script = requests.get(script_url).text

# Remove everything after the 'async def register_service' function
define_functions = script.split("async def register_service")[0]

# Set dummy variables
variables = """
WORKSPACE_TOKEN = "dummy"
CONTEXT = {"user": {"id": "dummy"}}
"""

# Define the execute function
run_segmentation_script = """
def execute(image, point_coordinates, point_labels):
    compute_embedding("vit_b", image, CONTEXT)
    features = segment(point_coordinates, point_labels, CONTEXT)
    return features
"""

script = variables + define_functions + run_segmentation_script
print(script)

# Define the pip requirements
base_requirements_file = os.path.join(BASE_URL, BRANCH, "requirements.txt")
base_requirements = requests.get(base_requirements_file).text

sam_requirements_file = os.path.join(BASE_URL, BRANCH, "requirements-sam.txt")
sam_requirements = requests.get(sam_requirements_file).text

pip_requirements = [
    requirement
    for requirement in (base_requirements + sam_requirements).split("\n")
    if requirement and not requirement.startswith(("#", "-r"))
]
print(pip_requirements)


async def main(name, script, pip_requirements):
    # Connect to the Hypha server
    server_url = "https://hypha.aicell.io"
    workspace_id = "bioengine-apps"
    service_id = "ray-function-registry"

    server = await connect_to_server({"server_url": server_url})

    # Retrieve the Ray Function Registry service
    svc = await server.get_service(f"{workspace_id}/{service_id}")

    # Register the ResNet function
    function_id = await svc.register_function(name=name, script=script, pip_requirements=pip_requirements)
    print(f"Registered function with id: {function_id}")

    # Example image
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_image.tif")
    image = imread(image_path)  # np.ndarray
    point_coordinates = [[10, 10]]  # Union[list, np.ndarray]
    point_labels = [0]  # Union[list, np.ndarray]

    # Run the ResNet function
    result = await svc.run_function(function_id=function_id, args=[image, point_coordinates, point_labels])
    print("Classification Result:", result)

if __name__ == "__main__":
    asyncio.run(main("microSAM", script, pip_requirements))