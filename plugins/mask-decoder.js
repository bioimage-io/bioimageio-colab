const loadSamDecoder = async ({ modelID }) => {
    const modelUrls = {
        "sam_vit_b_lm": "https://raw.githubusercontent.com/constantinpape/mbexc-review/refs/heads/master/vit_b_lm_decoder.onnx",
        "sam_vit_b_em_organelles": "https://raw.githubusercontent.com/constantinpape/mbexc-review/refs/heads/master/vit_b_em_decoder.onnx",
    };

    const modelUrl = modelUrls[modelID];
    console.log('Starting to created ONNX model from', modelUrl);
    const modelPromise = ort.InferenceSession.create(modelUrl)
        .then((model) => {
        console.log('ONNX model created', model);
        return model;
        })
        .catch((error) => {
        console.error('Error creating ONNX model:', error);
        throw error; // Propagate the error
        });

    return modelPromise;
};

const createTensorFromUint8Array = ({ data, shape }) => {
    const dst = new ArrayBuffer(data.byteLength);
    new Uint8Array(dst).set(new Uint8Array(data));
    return new ort.Tensor("float32", new Float32Array(dst), shape);
};

const computeEmbedding = async ({ samService, image, modelID }) => {
    // Compute the embedding for the image
    if (!samService) {
        const msg = "No SAM service available. Embedding computation was skipped.";
        console.log(msg);
        await api.showMessage(msg);
        return;
    }
  
    console.log(`Computing embedding for image with model ${modelID}...`);
    const embeddingPromise = samService.compute_embedding(image, modelID)
        .then(embeddingResult => {
            console.log("===== embeddingResult =====>", embeddingResult);
    
            // Set the model scale based on the image dimensions
            // TODO: check how to use embeddingResult.input_size (directly calculate modelScale)
            const longSideLength = Math.max(...embeddingResult["input_size"]);
            const w = image._rshape[0];
            const h = image._rshape[1];
            const samScale = longSideLength / Math.max(h, w);
            const modelScale = { height: h, width: w, samScale };
            console.log("===== modelScale =====>", modelScale);
    
            // Create the embedding tensor
            const embeddingTensor = createTensorFromUint8Array({
            data: embeddingResult.features._rvalue,
            shape: embeddingResult.features._rshape,
            });
            console.log("===== embeddingTensor =====>", embeddingTensor);
    
            return { embeddingTensor, modelScale };
        })
        .catch(error => {
            // Catch any errors during the embedding calculation or tensor preparation
            console.error('An error occurred while preparing the embedding:', error);
            throw error; // Propagate the error to be handled later
        });
  
    return embeddingPromise;
};

const prepareModelData = ({ embeddingResult, coordinates }) => {
    let pointCoords;
    let pointLabels;
    let pointCoordsTensor;
    let pointLabelsTensor;
  
    const embeddingTensor = embeddingResult.embeddingTensor;
    const modelScale = embeddingResult.modelScale;
  
    // Prepare input for the model
    const clicks = [ {
        x: coordinates[0], 
        y: coordinates[1], 
        clickType: 1
    } ];
  
    console.log('===== clicks =====>', clicks);
  
    // Check there are input click prompts
    if (clicks) {
        let n = clicks.length;

        // If there is no box input, a single padding point with 
        // label -1 and coordinates (0.0, 0.0) should be concatenated
        // so initialize the array to support (n + 1) points.
        pointCoords = new Float32Array(2 * (n + 1));
        pointLabels = new Float32Array(n + 1);

        // Add clicks and scale to what SAM expects
        for (let i = 0; i < n; i++) {
            pointCoords[2 * i] = clicks[i].x * modelScale.samScale;
            pointCoords[2 * i + 1] = clicks[i].y * modelScale.samScale;
            pointLabels[i] = clicks[i].clickType;
        }
  
        // Add in the extra point/label when only clicks and no box
        // The extra point is at (0, 0) with label -1
        pointCoords[2 * n] = 0.0;
        pointCoords[2 * n + 1] = 0.0;
        pointLabels[n] = -1.0;

        // Create the tensor
        pointCoordsTensor = new ort.Tensor("float32", pointCoords, [1, n + 1, 2]);
        pointLabelsTensor = new ort.Tensor("float32", pointLabels, [1, n + 1]);
    }
    const imageSizeTensor = new ort.Tensor("float32", [
        modelScale.height,
        modelScale.width,
    ]);
  
    if (pointCoordsTensor === undefined || pointLabelsTensor === undefined)
        return;
  
    // There is no previous mask, so default to an empty tensor
    const maskInput = new ort.Tensor(
        "float32",
        new Float32Array(256 * 256),
        [1, 1, 256, 256]
    );
    // There is no previous mask, so default to 0
    const hasMaskInput = new ort.Tensor("float32", [0]);
  
    const feeds = {
        image_embeddings: embeddingTensor,
        point_coords: pointCoordsTensor,
        point_labels: pointLabelsTensor,
        orig_im_size: imageSizeTensor,
        mask_input: maskInput,
        has_mask_input: hasMaskInput,
    };
    console.log('===== feeds =====>', feeds);
    return feeds;
};

const processMaskToGeoJSON = ({ masks }) => {
    // Dimensions of the mask (batch, channels, width, height)
    const [b, c, width, height] = masks.dims;
  
    // 1. Apply threshold to create binary mask
    const binaryMask = new Uint8Array(width * height);
    for (let i = 0; i < masks.data.length; i++) {
        binaryMask[i] = masks.data[i] > 0.0 ? 255 : 0;
    }
  
    // Convert binaryMask to an OpenCV.js Mat
    const binaryMat = new cv.Mat(height, width, cv.CV_8UC1);
    binaryMat.data.set(binaryMask);
  
    // 2. Use OpenCV.js to find contours
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(
        binaryMat,
        contours,
        hierarchy,
        cv.RETR_EXTERNAL, // Retrieve only external contours
        cv.CHAIN_APPROX_SIMPLE // Compress horizontal, vertical, and diagonal segments
    );
  
    // 3. Process contours into GeoJSON-compatible features
    const features = [];
  
    if (contours.size() > 0) {
        // Pick the largest contour as the main object
        let largestContourIndex = 0;
        let largestContourSize = 0;
        for (let i = 0; i < contours.size(); i++) {
        const c = contours.get(i);
        if (c.rows > largestContourSize) {
            largestContourSize = c.rows;
            largestContourIndex = i;
        }
        }

        const largestContour = contours.get(largestContourIndex);
        const pts = [];

        for (let i = 0; i < largestContour.rows; i++) {
            const x = largestContour.intPtr(i)[0]; // x coordinate
            const y = largestContour.intPtr(i)[1]; // y coordinate
            pts.push([x, y]);
        }

        // Close the polygon if not closed
        if (
            pts.length > 0 &&
            (pts[0][0] !== pts[pts.length - 1][0] || pts[0][1] !== pts[pts.length - 1][1])
        ) {
            pts.push(pts[0]);
        }

        // Add the polygon to the features array
        features.push(pts);
    }
  
    console.log('===== contours =====>', features);
  
    // Clean up memory
    contours.delete();
    hierarchy.delete();
    binaryMat.delete();
  
    return features;
};