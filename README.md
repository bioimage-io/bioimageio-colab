# BioImageIO Colab

This is a working-in-progress project to support **collaborative data annotation in the browser using the BioEngine and Kaibu**. It allows the safe dissemination of images embedded in an image annotation tool (Kaibu) and storing the corresponding annotation in a source directory. This functionality is enabled by the connection to the BioEngine server. 

### Collaboratory annotation in the browser
- To test a demo you can open the [BioImage.IO Crow Sourcing](https://github.com/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb) notebook in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing.ipynb).
- To test a demo connected to cellpose model fine-tuning, open [BioImage.IO Crow Sourcing and Cellpose](https://github.com/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing_CellposeFinetune.ipynb) notebook in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bioimage-io/bioimageio-colab/blob/main/notebooks/BioImageIOCrowdSourcing_CellposeFinetune.ipynb).

### Collaboratory annotation powered by Segment Anything Model (SAM)

We support using segment-anything model (SAM) to facilitate the annotation of images for segmentation. To try it, install the following Python dependencies:

```
conda create -n bioimageio-colab python=3.11
conda activate bioimageio-colab
pip install -r requirements.txt
```
and run the BioEngine server setup for this:
```bash

python -m bioimageio_colab.sam_annotation_server
```

Then go to the printed URL and start collaborative annotation.

### Building your own collaboratory annotation interface

BioImageIO Colab is based on two main components: Kaibu (a web-browser annotation tool) and the BioEngine. The BioEngine provides a framework in which different clients (the "host" and the "collaborators") are connected to a common server (the BioEngine server) in a way that makes it possible to transfer data and interfaces without installation. 

Therefore, these are the components to consider when designing your collaborator annotation tool: 
- **The information you want to send and retrieve in the process**: For example, the images you want to send to other clients through the server and the information you want to recover after a client annotates the images. This is defined int he `service` defined in Python and determined by the functions in this service. It has the following appearance:

   ```python
    svc = await server.register_service({
    "name": "Model Trainer",
    "id": "bioimageio-colab",
    "config": {
        "visibility": "public"
    },
    "get_random_image": get_random_image, # get_random_image provides a numpy array with an image
    "save_annotation": save_annotation, # save_annotation saves a given annotation in a given directory
    })
    ```
- The **interaction ImJoy plugin that runs in the BioEngine server** and takes care of the input provided by the host client and the input from the collaborators. This plugin is an `html` file that follows a template similar to [this one](https://github.com/bioimage-io/bioimageio-colab/blob/main/plugins/bioimageio-colab.imjoy.html).
  -  Importantly, the `BioImageIOColabAnnotator` class is the one that determines the interaction and the appearance of Kaibu's interface and for optimization purposes, it is recommended to write it in Java. 
Still, the script is simple. Consider customising the following two functions to personalise the interface in Kaibu or the images displayed there:
      ```python
      // Define image reading and displaying function
      const getImage = async () => {
          if (this.image !== null) {
              await viewer.remove_layer({id: this.imageLayer.id});
              await viewer.remove_layer({id: this.annotationLayer.id});
          }
  
          [this.image, this.filename, this.newname] = await this.biocolab.get_random_image();
          this.imageLayer = await viewer.view_image(this.image, {name: "image"});
          // Add the annotation functionality to the interface
          this.annotationLayer = await viewer.add_shapes([], {
              shape_tpe: "polygon",
              draw_edge_color: "magenta",
              name: "annotation",
          });
        };
  
      const saveAnnotation = async () => {
          if(!this.annotationLayer) return;
          const annotation = await this.annotationLayer.get_features();
          if(annotation.features.length > 0){
              await this.biocolab.save_annotation(this.filename, this.newname, annotation, [this.image._rshape[0], this.image._rshape[1]]);
              await api.showMessage("Annotation Saved to " + this.filename);
          }
          else{
              await api.showMessage("Skip saving annotation");
          }
          
      };
      ```
