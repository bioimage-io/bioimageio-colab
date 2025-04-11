# Use an official slim Python 3.11 image
FROM python:3.11.9-slim

# Set the working directory
WORKDIR /app/

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with sudo privileges
RUN groupadd -r bioimageio_colab && useradd -r -g bioimageio_colab --create-home bioimageio_colab \
    && usermod -aG sudo bioimageio_colab \
    && echo "bioimageio_colab ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/bioimageio_colab

# Copy requirements first for efficient layer caching
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies with --no-cache-dir to reduce image size
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application files last to leverage Docker's build cache
COPY . /app/

# Change ownership to the non-root user and ensure scripts are executable
RUN chown -R bioimageio_colab:bioimageio_colab /app/ \
    && chmod +x /app/scripts/*

# Switch to the non-root user
USER bioimageio_colab

# Default entrypoint for running the application
CMD ["bash"]
