# BioImage.IO Colab

This is a work-in-progress project to support **collaborative data annotation in the browser using the BioEngine and Kaibu**. It allows the safe dissemination of images embedded in an image annotation tool (Kaibu) and storing the corresponding annotation in a source directory. This functionality is enabled by the connection to the BioEngine server.

### 🚀 Try the Interactive Demo!

Experience the power of the Segment Anything Model (SAM) in your browser: [Launch Demo](https://imjoy.io/lite?plugin=https://raw.githubusercontent.com/bioimage-io/bioimageio-colab/refs/heads/main/plugins/bioimageio-colab-annotator.imjoy.html). This demo showcases interactive segmentation with SAM on an example image.

For collaborative annotation sessions with your own data and the ability to save annotations, visit our [Web Interface](https://bioimage-io.github.io/bioimageio-colab/).

### Components

BioImageIO Colab combines two powerful tools:
- **Kaibu**: A web-based annotation tool
- **BioEngine**: A server framework that connects "Data Provider" and "Collaborators"

#### Key Components for Collaborative Annotation

1. **Data Flow Configuration**
   - Images are sent from Data Provider to Collaborators
   - Annotations are sent back to Data Provider
   - Two main functions: `get_random_image` and `save_annotation`
   - Set up through [Web Interface](https://bioimage-io.github.io/bioimageio-colab/)
   - Implementation in [docs/data_providing_service.py](https://github.com/bioimage-io/bioimageio-colab/blob/main/docs/data_providing_service.py)

2. **Interactive Interface**
   - ImJoy plugin handles image display and annotation
   - Connects to Data Provider service
   - Available as [plugin](https://github.com/bioimage-io/bioimageio-colab/blob/main/plugins/bioimageio-colab-annotator.imjoy.html)

### Integrate SAM Compute Service

#### Required Dependencies
```html
<!-- Hypha RPC WebSocket -->
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.47/dist/hypha-rpc-websocket.min.js">

<!-- ONNX Runtime Web -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js">

<!-- OpenCV -->
<script src="https://docs.opencv.org/4.5.0/opencv.js">

<!-- SAM Mask Decoder -->
<script src="https://cdn.jsdelivr.net/gh/bioimage-io/bioimageio-colab@latest/plugins/onnx-mask-decoder.js">
```

#### Integration Steps

1. **Connect to BioEngine**
```javascript
const client = await hyphaWebsocketClient.connectToServer({
    server_url: "https://hypha.aicell.io",
});
samService = await client.getService("bioimageio-colab/microsam", { mode: "last" });
dataService = await client.getService("<your_data_service_id>");
```

2. **Load SAM Model**
```javascript
const modelID = "sam_vit_b_lm"; // or "sam_vit_b_em_organelles"
modelPromise = loadSamDecoder({ modelID: modelID });
```

3. **Process Images**
```javascript
// Load image
const image = await dataService.get_random_image();

// Compute embedding (once per image)
embeddingPromise = computeEmbedding({
    samService: samService,
    image: image,
    modelID: modelID,
});

// Segment with point prompt
const coordinates = [63, 104];
const results = await segmentImage({
    model: modelPromise,
    embedding: embeddingPromise,
    coordinates: coordinates,
});
const mask = results["masks"];

// Convert to GeoJSON format
const polygonCoords = processMaskToGeoJSON({
    masks: mask,
    threshold: 0,
});
```