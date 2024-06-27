# Use an image with Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

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

# Copy the bioimageio-colab folder to the docker environment
COPY . /app/bioimageio-colab/.

# Install bioimageio_colab_server as a module
RUN pip install -e bioimageio-colab

# Run the bioimageio_colab_server model server
ENTRYPOINT ["python", "-m", "bioimageio_colab.segmentation_server"]
