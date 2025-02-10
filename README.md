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

The SAM compute service can be integrated both in JavaScript (browser) and Python environments.

#### JavaScript Integration (Browser)

##### Required Dependencies
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

##### Integration Steps

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

#### Python Integration

##### Required Dependencies
```bash
pip install opencv-python
pip install onnxruntime
pip install hypha-rpc
```

##### Integration Steps

1. **Connect to BioEngine**
```python
from hypha_rpc import connect_to_server

client = await connect_to_server({"server_url": "https://hypha.aicell.io"})
svc = await client.get_service("bioimageio-colab/microsam", {"mode": "last"})
```

2. **Load SAM Model**
```python
model_id = "sam_vit_b_lm"  # or "sam_vit_b_em_organelles"
model = load_sam_decoder(model_id)
```

3. **Process Images**
```python
# Load and process image
image = load_image("path/to/image.tif")

# Compute embedding
embedding_result = await svc.compute_embedding(
    image=image,
    model_id=model_id,
)

# Segment with point prompt
example_coordinates = (80, 80)
feeds = prepare_model_data(embedding_result, example_coordinates)
masks = model.run(["masks"], feeds)

# Process mask (example: convert to binary and find contours)
mask = masks[0].squeeze()
binary_mask = (mask > 0).astype(np.uint8)
contours, _ = cv2.findContours(
    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
```

For complete implementations, see:
- JavaScript: [plugins/onnx-mask-decoder.js](plugins/onnx-mask-decoder.js)
- Python: [bioimageio_colab/onnx_mask_decoder.py](bioimageio_colab/onnx_mask_decoder.py)