# Use an image with Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app/

# Create a non-root user
RUN groupadd -r bioimageio_colab && useradd -r -g bioimageio_colab bioimageio_colab

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file for SAM to the docker environment
COPY ./requirements-sam.txt /app/requirements-sam.txt

# Install the required packages for SAM
RUN pip install -r /app/requirements-sam.txt

# Copy the requirements file to the docker environment
COPY ./requirements.txt /app/requirements.txt

# Install the required Python packages
RUN pip install -r /app/requirements.txt

# Copy the python script to the docker environment
COPY ./bioimageio_colab/register_sam_service.py /app/register_sam_service.py

# Change ownership of the application directory to the non-root user
RUN chown -R bioimageio_colab:bioimageio_colab /app/

# Switch to the non-root user
USER bioimageio_colab

# Register the segmentation model as a hypha service
ENTRYPOINT ["python", "register_sam_service.py"]
