import os

from hypha_rpc.hypha.sync import login


def main():
    # Login to hypha
    token = login({"server_url": "https://hypha.aicell.io"})

    # Save the token to a .env file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    
    updated_env_lines = []
    found_token_line = False
    
    # Read existing .env file if it exists
    if os.path.exists(file_path):
        with open(file_path, "r") as env_file:
            for line in env_file:
                if line.strip().startswith("HYPHA_TOKEN="):
                    updated_env_lines.append(f'HYPHA_TOKEN="{token}"\n')
                    found_token_line = True
                else:
                    updated_env_lines.append(line)
    
    # If HYPHA_TOKEN line wasn't found, add it
    if not found_token_line:
        updated_env_lines.append(f'HYPHA_TOKEN="{token}"\n')
    
    # Write back to the .env file
    with open(file_path, "w") as env_file:
        env_file.writelines(updated_env_lines)

    # Define the permission mode for read and write for the owner only
    mode = 0o600

    # Change the file permissions
    os.chmod(file_path, mode)

if __name__ == "__main__":
    main()
