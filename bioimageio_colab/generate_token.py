"""
This script generates a workspace token and saves it to the .env file.

User interface for this feature available in the Hypha dashboard ( https://hypha.aicell.io/bioimageio-colab#development -> "Generate Token" button ).
"""

import argparse
import os

import requests
from hypha_rpc.sync import login


def generate_token(server_url, workspace, expires_in):
    generate_token_url = f"{server_url}/public/services/ws/generate_token"
    token = login({"server_url": server_url})

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    payload = {"config": {"expires_in": expires_in, "workspace": workspace}}

    response = requests.post(generate_token_url, json=payload, headers=headers)

    if response.ok:
        workspace_token = response.json()

        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        env_file = os.path.join(base_dir, ".env")
        # Read existing .env file into a dictionary
        with open(env_file, "r") as f:
            env_vars = dict(
                tuple(line.strip().split("=")) for line in f if not line.strip().startswith("#")
            )
        # Replace the workspace token
        env_vars["WORKSPACE_TOKEN"] = workspace_token
        # Write the dictionary back to the .env file
        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        print(f"Token generated successfully and saved to {env_file}")

        return workspace_token
    else:
        raise Exception(
            f"Failed to generate token. Please try again later. ({response.status_code}: {response.reason})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a workspace token")
    parser.add_argument(
        "--server_url", default="https://hypha.aicell.io", type=str, help="Server URL"
    )
    parser.add_argument(
        "--workspace", default="bioimageio-colab", type=str, help="Workspace name"
    )
    parser.add_argument(
        "--expires_in", default=3600, type=int, help="Token expiration time in seconds"
    )

    args = parser.parse_args()

    token_data = generate_token(args.server_url, args.workspace, args.expires_in)
