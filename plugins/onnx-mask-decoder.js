const loadSamDecoder = async ({ modelID }) => {
    const modelUrls = {
        "sam_vit_b_lm": "https://raw.githubusercontent.com/constantinpape/mbexc-review/refs/heads/master/vit_b_lm_decoder.onnx",
        "sam_vit_b_em_organelles": "https://raw.githubusercontent.com/constantinpape/mbexc-review/refs/heads/master/vit_b_em_decoder.onnx",
    };

    const modelUrl = modelUrls[modelID];
    console.log("Starting to created ONNX model from", modelUrl);
    const modelPromise = ort.InferenceSession.create(modelUrl)
        .then((model) => {
            console.log("ONNX model created", model);
            return model;
        })
        .catch((error) => {
            console.error("Error creating ONNX model:", error);
            throw error; // Propagate the error
        });

    return modelPromise;
};

const createInputTensors = ({ embeddingFeatures, originalImageShape, samScale }) => {
    // Check the embedding features
    if (embeddingFeatures._rdtype !== "float32") {
        throw new Error(`Invalid embedding features data type. Expected 'float32', but received '${embeddingFeatures._rdtype}'.`);
    }

    if (!(embeddingFeatures._rvalue instanceof Uint8Array)) {
        throw new Error(`Invalid embedding features value. Expected an instance of Uint8Array, but received '${typeof embeddingFeatures._rvalue}'.`);
    }

    if (embeddingFeatures._rshape.length !== 4) {
        throw new Error(`Invalid embedding features shape. Expected a 4-element array, but received '${embeddingFeatures._rshape.length}' elements.`);
    }

    const nEmbeddingElements = embeddingFeatures._rshape.reduce(
        (accumulator, currentValue) => accumulator * currentValue, 1
    );
    if (embeddingFeatures._rvalue.byteLength !== nEmbeddingElements * 4) {
        throw new Error(
            `Mismatch in embedding size. Expected ${nEmbeddingElements * 4} bytes (for ${nEmbeddingElements} float32 elements), ` +
            `but received ${embeddingFeatures._rvalue.byteLength} bytes.`
        );
    }

    // Copy the embedding features to a new buffer to reset the byte offset
    const newBuffer = new ArrayBuffer(embeddingFeatures._rvalue.byteLength);
    new Uint8Array(newBuffer).set(embeddingFeatures._rvalue);

    // Create the embedding tensor
    const embeddingTensor = new ort.Tensor(
        "float32",
        new Float32Array(newBuffer),
        embeddingFeatures._rshape
    );

    // Check the original image shape
    if (originalImageShape.length !== 2) {
        throw new Error(`Invalid original image shape. Expected a 2-element array, but received '${originalImageShape.length}' elements.`);
    }

    // Create the image size tensor
    const imageSizeTensor = new ort.Tensor(
        "float32",
        [originalImageShape[0], originalImageShape[1]],
    );

    // Check the SAM scale
    if (typeof samScale !== "number" || isNaN(samScale) || samScale <= 0) {
        throw new Error(`Invalid SAM scale. Expected a positive number, but received '${samScale}'.`);
    }

    // There is no previous mask, so default to an empty tensor
    const maskInput = new ort.Tensor(
        "float32",
        new Float32Array(256 * 256),
        [1, 1, 256, 256],
    );
    // There is no previous mask, so default to 0
    const hasMaskInput = new ort.Tensor("float32", [0]);

    return { embeddingTensor, imageSizeTensor, samScale, maskInput, hasMaskInput };
};

const computeEmbedding = async ({ samService, image, modelID }) => {
    // Compute the embedding for the image
    console.log(`Computing embedding for image with model ${modelID}...`);
    const embeddingPromise = samService.compute_embedding(image, modelID)
        .then(embeddingResult => {
            console.log("Received embedding result:", embeddingResult);
            const inputTensors = createInputTensors({
                embeddingFeatures: embeddingResult["features"],
                originalImageShape: embeddingResult["original_image_shape"],
                samScale: embeddingResult["sam_scale"],
            });
            console.log("Input tensors created:", inputTensors);
            return inputTensors;
        })
        .catch(error => {
            // Catch any errors during the embedding calculation or tensor preparation
            console.error("An error occurred while preparing the embedding:", error);
            throw error; // Propagate the error to be handled later
        });

    return embeddingPromise;
};

const prepareModelData = ({ embeddingResult, coordinates }) => {
    // Check the coordinates
    if (coordinates.length !== 2) {
        console.error("Invalid coordinates. Expected a 2-element array, but received", coordinates);
        return;
    }
    if (typeof coordinates[0] !== "number" || isNaN(coordinates[0]) || coordinates[0] < 0) {
        console.error("Invalid x-coordinate. Expected a non-negative number, but received", coordinates[0]);
        return;
    }
    if (typeof coordinates[1] !== "number" || isNaN(coordinates[1]) || coordinates[1] < 0) {
        console.error("Invalid y-coordinate. Expected a non-negative number, but received", coordinates[1]);
        return;
    }

    // Create click points
    const clicks = [{
        x: coordinates[0],
        y: coordinates[1],
        clickType: 1
    }];

    // Check there are input click prompts
    let n = clicks.length;

    // If there is no box input, a single padding point with 
    // label -1 and coordinates (0.0, 0.0) should be concatenated
    // so initialize the array to support (n + 1) points.
    const pointCoords = new Float32Array(2 * (n + 1));
    const pointLabels = new Float32Array(n + 1);

    // Add clicks and scale to what SAM expects
    for (let i = 0; i < n; i++) {
        pointCoords[2 * i] = clicks[i].x * embeddingResult.samScale;
        pointCoords[2 * i + 1] = clicks[i].y * embeddingResult.samScale;
        pointLabels[i] = clicks[i].clickType;
    }

    // Add in the extra point/label when only clicks and no box
    // The extra point is at (0, 0) with label -1
    pointCoords[2 * n] = 0.0;
    pointCoords[2 * n + 1] = 0.0;
    pointLabels[n] = -1.0;

    // Create the tensor
    const pointCoordsTensor = new ort.Tensor("float32", pointCoords, [1, n + 1, 2]);
    const pointLabelsTensor = new ort.Tensor("float32", pointLabels, [1, n + 1]);

    return {
        image_embeddings: embeddingResult.embeddingTensor,
        point_coords: pointCoordsTensor,
        point_labels: pointLabelsTensor,
        orig_im_size: embeddingResult.imageSizeTensor,
        mask_input: embeddingResult.maskInput,
        has_mask_input: embeddingResult.hasMaskInput,
    };
};

const segmentImage = async ({ model, embedding, coordinates }) => {
    const embeddingResult = await embedding;
    const feeds = prepareModelData({
        embeddingResult: embeddingResult,
        coordinates: coordinates,
    });
    console.log("Feeds prepared for the model:", feeds);
    const modelLoaded = await model;
    const results = await modelLoaded.run(feeds);
    console.log("Model results:", results);
    return results;
};

const processMaskToGeoJSON = ({ masks, threshold = 0 }) => {
    // Dimensions of the mask (batch, channels, width, height)
    const [b, c, width, height] = masks.dims;

    // 1. Apply threshold to create binary mask
    const binaryMask = new Uint8Array(width * height);
    for (let i = 0; i < masks.data.length; i++) {
        binaryMask[i] = masks.data[i] > threshold ? 255 : 0;
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

    console.log("Features extracted from the mask:", features);

    // Clean up memory
    contours.delete();
    hierarchy.delete();
    binaryMat.delete();

    return features;
};