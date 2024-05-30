# BioImageIO Colab

This is a working-in-progress project to support **collaborative data annotation in the browser using the BioEngine and Kaibu**. It allows safe dissemination of images embedded in an image annotation tool (Kaibu) and storing the corresponding annotation in a source directory. This functionality is enabled by the connection to the BioEngine server. 

- To test a demo you can open the [BioImage.IO Crow Sourcing](https://github.com/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb) notebook in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb).
- To test a demo connected to cellpose model fine-tuning, open [BioImage.IO Crow Sourcing and Cellpose](https://github.com/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing_CellposeFinetune.ipynb) notebook in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing_CellposeFinetune.ipynb).

### Colab annotation powered by Segment Anything Model (SAM)

We support using segment-anything model (SAM) to facilitate the annotation of images for segmentation. To try it, install the following Python dependencies:

```
conda create -n bioimageio-colab python=3.11
conda activate bioimageio-colab
pip install -r requirements.txt
```
and run the BioEngine server setup for this:
```bash

python -m bioimageio_colab.sam_server
```

Then go to https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/main/plugins/bioimageio-colab-sam.imjoy.html to start interactive annotation.
