import os
import argparse
from hypha_rpc import connect_to_server, login
import asyncio


async def create_workspace_token(args):
    # Get a user login token
    token = await login({"server_url": args.server_url})

    # Connect to the Hypha server
    server = await connect_to_server(
        {
            "server_url": args.server_url,
            "token": token,
        }
    )

    # Check if the workspace already exists
    user_workspaces = await server.list_workspaces()
    exists = any(
        [args.workspace_name == workspace["name"] for workspace in user_workspaces]
    )

    # Create a workspace
    if not exists or args.overwrite:
        workspace = await server.create_workspace(
            {
                "name": args.workspace_name,
                "description": args.description,
                "owners": args.owners,
                "allow_list": args.allow_list,
                "deny_list": args.deny_list,
                "visibility": "public",  # public/protected
                "persistent": True,  # keeps the workspace alive even after a server restart
            },
            overwrite=args.overwrite,
        )
        # Check if the workspace was created
        assert any(
            [
                args.workspace_name == workspace["name"]
                for workspace in await server.list_workspaces()
            ]
        )
        print(f"Workspace created: {workspace['name']}")

    # Generate a workspace token
    token = await server.generate_token(
        {
            "workspace": args.workspace_name,
            "expires_in": args.token_expires_in,
            "permission": args.token_permission,
            # "extra_scopes": [],
        }
    )
    expires_in_days = args.token_expires_in / 60 / 60 / 24
    print(
        f"Workspace token generated:\n - Permission: '{args.token_permission}'\n - Expires in: {expires_in_days} days"
    )

    # Save the token to a .env file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

    # Read existing .env file if it exists
    updated_env_lines = []
    found_token_line = False
    if os.path.exists(file_path):
        with open(file_path, "r") as env_file:
            for line in env_file:
                if line.strip().startswith("WORKSPACE_TOKEN="):
                    updated_env_lines.append(f'WORKSPACE_TOKEN="{token}"\n')
                    found_token_line = True
                else:
                    updated_env_lines.append(line)

    # If WORKSPACE_TOKEN line wasn't found, add it
    if not found_token_line:
        updated_env_lines.append(f'WORKSPACE_TOKEN="{token}"\n')

    # Write back to the .env file
    with open(file_path, "w") as env_file:
        env_file.writelines(updated_env_lines)

    # Define the permission mode for read and write for the owner only
    mode = 0o600

    # Change the file permissions
    os.chmod(file_path, mode)

    print(f"Token saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the BioImageIO Colab workspace."
    )
    parser.add_argument(
        "--server_url",
        default="https://hypha.aicell.io",
        type=str,
        help="URL of the Hypha server",
    )
    parser.add_argument(
        "--workspace_name",
        default="bioimageio-colab",
        type=str,
        help="Name of the workspace",
    )
    parser.add_argument(
        "--description",
        default="The BioImageIO Colab workspace for serving interactive segmentation models.",
        type=str,
        help="Description of the workspace",
    )
    parser.add_argument(
        "--owners",
        nargs="+",
        default=[],
        type=str,
        help="User emails that own the workspace",  # user email of workspace creator is added automatically
    )
    parser.add_argument(
        "--allow_list",
        nargs="+",
        default=[],
        type=str,
        help="User emails allowed access to the workspace",
    )
    parser.add_argument(
        "--deny_list",
        nargs="+",
        default=[],
        type=str,
        help="User emails denied access to the workspace",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the workspace if it already exists",
    )
    parser.add_argument(
        "--token_expires_in",
        default=31536000,
        type=int,
        help="Token expiration time in seconds (default: 1 year)",
    )
    parser.add_argument(
        "--token_permission",
        default="read_write",
        type=str,
        help="Token permission (must be one of: read, read_write, admin)",
    )

    args = parser.parse_args()
    asyncio.run(create_workspace_token(args))
