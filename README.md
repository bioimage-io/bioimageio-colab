# BioImageIO Colab

This is a working-in-progress project to support collaborative data annotation in the browser using the BioEngine and Kaibu. It allows to safely disseminate images embedded in an image annotation tool (Kaibu) and storing the corresponding annotation in a source directory. This functionality is enabled by the connection to the BioEngine server. 

- To test a demo you can open the [BioImage.IO Crow Sourcing](https://github.com/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb) notebook in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb).
- To test a demo connected to Cellpose model fine-tuning, open [BioImage.IO Crow Sourcing and Cellpose](https://github.com/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing_CellposeFinetune.ipynb) notebook in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing_CellposeFinetune.ipynb).

## Segment Anything

We support using segment-anything models to facilitate annotation.

```
conda create -n bioimageio-colab python=3.11
conda activate bioimageio-colab
pip install -r requirements.txt
```

```bash

python -m bioimageio_colab.sam_server
```

Then go to https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/main/plugins/bioimageio-colab-sam.imjoy.html to start interactive annotation.
