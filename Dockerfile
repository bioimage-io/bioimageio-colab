# Use an image with Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app/

# Create a non-root user
RUN groupadd -r bioimageio_colab && useradd -r -g bioimageio_colab bioimageio_colab

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file to the docker environment
COPY ./bioimageio_colab/requirements.txt /app/requirements.txt

# Install the required Python packages
RUN pip install \
    -r /app/requirements.txt \
    segment_anything \
    torch \
    torchvision

# Copy the python script to the docker environment
COPY ./bioimageio_colab/segmentation_model.py /app/segmentation_model.py

# Change ownership of the application directory to the non-root user
RUN chown -R bioimageio_colab:bioimageio_colab /app/

# Switch to the non-root user
USER bioimageio_colab

# Register the segmentation model as a hypha service
ENTRYPOINT ["python", "segmentation_model.py"]
