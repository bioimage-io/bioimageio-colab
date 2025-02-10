# BioImage.IO Colab

This is a work-in-progress project to support **collaborative data annotation in the browser using the BioEngine and Kaibu**. It allows the safe dissemination of images embedded in an image annotation tool (Kaibu) and storing the corresponding annotation in a source directory. This functionality is enabled by the connection to the BioEngine server.

### Collaborative annotation powered by Segment Anything Model (SAM)

We support using the Segment Anything Model (SAM) to facilitate the annotation of images for segmentation. To try it, open the [demo](https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/refs/heads/main/plugins/bioimageio-colab-annotator.imjoy.html) with an example image.

### Components

BioImageIO Colab is based on two main components: Kaibu (a web-browser annotation tool) and the BioEngine. The BioEngine provides a framework in which different clients (the "Data Provider" and the "Collaborators") are connected to a common server (the BioEngine server) in a way that makes it possible to transfer data and interfaces without installation.

These are the components to consider when designing your collaborative annotation:
- **The information you want to send and retrieve in the process**: For example, the images you want to send to other clients through the server and the information you want to recover after a client annotates the images.
  The "Data Provider" registers two functions to the BioEngine, one to load a random image from the provided list of images (`get_random_image`) and another to save the annotation back to the "Data Provider" (`save_annotation`). These functions are defined in [docs/data_providing_service.py](https://github.com/bioimage-io/bioimageio-colab/blob/main/docs/data_providing_service.py). This can be done through the web UI at [bioimage-io.github.io/bioimageio-colab/](https://bioimage-io.github.io/bioimageio-colab/).

- The **interactive ImJoy plugin** takes care of the images provided by the host client and the input from the "Collaborators". This [plugin](https://github.com/bioimage-io/bioimageio-colab/blob/main/plugins/bioimageio-colab-annotator.imjoy.html) is an `html` file and can be loaded using [imjoy.io](https://imjoy.io/lite). A link with a configuration to connect to the "Data Provider" service is returned in the [web UI](https://bioimage-io.github.io/bioimageio-colab/).

### Integrate the SAM compute service into your own project

Load these requirements:
```
<!-- Hypha RPC WebSocket -->
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.47/dist/hypha-rpc-websocket.min.js">

<!-- ONNX Runtime Web -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js">

<!-- OpenCV -->
<script src="https://docs.opencv.org/4.5.0/opencv.js">

<!-- SAM Mask Decoder -->
<script src="https://cdn.jsdelivr.net/gh/bioimage-io/bioimageio-colab@latest/plugins/onnx-mask-decoder.js">
```

Connect to the BioEngine:
```javascript
const client = await hyphaWebsocketClient.connectToServer({
    server_url: "https://hypha.aicell.io",
});
samService = await client.getService("bioimageio-colab/microsam", { mode: "last" });
dataService = await client.getService("<your_data_service_id>");
```

Load the SAM decoder as an ONNX model. Currently, `sam_vit_b_lm` and `sam_vit_b_em_organelles` are available models:
```javascript
const modelID = "sam_vit_b_lm";
modelPromise = loadSamDecoder({ modelID: modelID });
```
Working with promises allows the model to be loaded in the background. Simply `await` the promise to get the ONNX model.

Load an image from the data service:
```javascript
const image = await dataService.get_random_image();
```

Compute the image embedding using the provided SAM compute service. This only needs to be done once for every new image:
```javascript
embeddingPromise = computeEmbedding({
    samService: samService,
    image: image,
    modelID: modelID,
});
```

Given a prompt with point coordinates, `segmentImage` will use the SAM mask decoder and the image embedding to segment the image at the given position. `segmentImage` can handle both pending and awaited promises:
```javascript
const coordinates = [63, 104];
const results = await segmentImage({
    model: modelPromise,
    embedding: embeddingPromise,
    coordinates: coordinates,
});
const mask = results["masks"];
```

The received mask still needs to be processed in order to be used in Kaibu. `processMaskToGeoJSON` first applies a threshold to create a binary mask and then uses OpenCV to find contours and extract x and y coordinates of a closed polygon:
```javascript
const polygonCoords = processMaskToGeoJSON({
    masks: mask,
    threshold: 0,
});
```
The threshold can be adjusted as needed.