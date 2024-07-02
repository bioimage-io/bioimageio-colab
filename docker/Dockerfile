# Use an image with Python 3.11
FROM python:3.11-slim

# Create a non-root user
RUN groupadd -r bioimageio_colab && useradd -r -g bioimageio_colab bioimageio_colab

# Upgrade pip
RUN pip install --upgrade pip

# Install the required Python packages
RUN pip install numpy==1.26.4 \
    imjoy-rpc==0.5.48.post2 \
    requests==2.31.0 \
    kaibu-utils==0.1.14 \
    segment_anything \
    torch \
    torchvision

# Set the working directory
WORKDIR /app/

# Copy the python script to the docker environment
COPY ./bioimageio_colab/segmentation_model.py /app/segmentation_model.py

# Change ownership of the application directory to the non-root user
RUN chown -R bioimageio_colab:bioimageio_colab /app/

# Switch to the non-root user
USER bioimageio_colab

# Register the segmentation model as a hypha service
# Arguments:
# --token <token>: Token for connecting to the Hypha server
# --client_id <client_id>: Client ID for registering the service. Default is "model-server"
# --service_id <service_id>: Service ID for registering the service. Default is "interactive-segmentation"
ENTRYPOINT ["python", "segmentation_model.py"]

LABEL org.opencontainers.image.source https://github.com/bioimage-io/bioimageio-colab
