# BioImageIO Colab


Open the [BioImage.IO Crow Sourcing](https://github.com/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb) notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb)


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
