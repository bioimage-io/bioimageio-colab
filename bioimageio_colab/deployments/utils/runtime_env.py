from pathlib import Path
from typing import List


def parse_requirements(file_path: Path) -> List[str]:
    """Parse a requirements file and return a list of packages"""
    requirements_file = Path(file_path).absolute()
    # Read the requirements file
    text = requirements_file.read_text()
    # Filter and clean package names (skip comments and empty lines)
    skip_lines = ("#", "-r ")
    packages = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith(skip_lines)
    ]
    return packages


if __name__ == "__main__":
    # Example usage
    requirements_file = Path("requirements-sam.txt")
    runtime_env = {"pip": parse_requirements(requirements_file)}
    print(runtime_env)
