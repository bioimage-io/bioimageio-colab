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
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements-sam.txt /app/requirements-sam.txt

# Install the required packages for SAM
RUN pip install -r /app/requirements-sam.txt

# Copy the python script to the docker environment
COPY ./bioimageio_colab/register_sam_service.py /app/register_sam_service.py

# Copy the start service script
COPY ./start_service.sh /app/start_service.sh

# Change ownership of the application directory to the non-root user
RUN chown -R bioimageio_colab:bioimageio_colab /app/

# Make the start script executable
RUN chmod +x /app/start_service.sh

# Switch to the non-root user
USER bioimageio_colab

# Use the start script as the entrypoint and forward arguments
ENTRYPOINT ["/app/start_service.sh"]
