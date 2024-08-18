# Use an image with Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app/

# Create a non-root user
RUN groupadd -r bioimageio_colab && useradd -r -g bioimageio_colab bioimageio_colab

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file for SAM to the docker environment
COPY ./requirements-sam.txt /app/requirements-sam.txt

# Install the required packages for SAM
RUN pip install -r /app/requirements-sam.txt

# Fetch the Hypha server version
RUN HYPHA_VERSION=$(curl -s https://hypha.aicell.io/config.json | jq -r '.hypha_version')

# Add a label to the docker image
LABEL hypha_version=$HYPHA_VERSION

# Reinstall or upgrade hypha-rpc to the version matching the Hypha server
RUN pip install --upgrade "hypha-rpc<=$HYPHA_VERSION"

# Copy the python script to the docker environment
COPY ./bioimageio_colab/register_sam_service.py /app/register_sam_service.py

# Change ownership of the application directory to the non-root user
RUN chown -R bioimageio_colab:bioimageio_colab /app/

# Switch to the non-root user
USER bioimageio_colab

# Register the segmentation model as a hypha service
ENTRYPOINT ["python", "register_sam_service.py"]
