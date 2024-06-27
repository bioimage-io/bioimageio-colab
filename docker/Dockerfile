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

# Copy the bioimageio-colab folder to the docker environment
COPY . /app/bioimageio-colab/.

# Set the working directory
WORKDIR /app/bioimageio-colab/

# Install bioimageio_colab as a module
RUN pip install -e .

# Run the bioimageio_colab model service
ENTRYPOINT ["python", "-m", "bioimageio_colab.segmentation_model"]
