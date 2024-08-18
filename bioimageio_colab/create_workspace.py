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
    else:
        print(f"Workspace already exists: {args.workspace_name}")

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

    args = parser.parse_args()
    asyncio.run(create_workspace_token(args))
