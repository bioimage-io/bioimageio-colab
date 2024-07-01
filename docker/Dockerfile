# Use an image with Python 3.11
FROM python:3.11-slim

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

# Register the segmentation model as a hypha service
ENTRYPOINT ["python", "segmentation_model.py"]
