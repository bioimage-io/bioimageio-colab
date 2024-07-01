import os

from imjoy_rpc.hypha.sync import login


def main():
    # Login to hypha
    token = login({"server_url": "https://ai.imjoy.io"})

    # Save the token to a .env file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    with open(file_path, "w") as env_file:
        env_file.write(f'HYPHA_TOKEN="{token}"')

    # Define the permission mode for read and write for the owner only
    mode = 0o600

    # Change the file permissions
    os.chmod(file_path, mode)

if __name__ == "__main__":
    main()