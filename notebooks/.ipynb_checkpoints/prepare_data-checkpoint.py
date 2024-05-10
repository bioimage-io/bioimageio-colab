from tifffile import imread, imsave
import numpy as np
import os


path2data = "/Users/esti/Documents/PROYECTOS/BIOIMAGEIO/bioimageio-colab/data/u2os-leica-collective/"
outpath = "/Users/esti/Documents/PROYECTOS/BIOIMAGEIO/bioimageio-colab/data/u2os-leica-collective-cropped"
os.makedirs(outpath, exist_ok=True)
import cv2
files = [f for f in os.listdir(path2data) if f.endswith(".tif")]
size = 512
resize = 0.25
for f in files:
    k = 0
    im = imread(os.path.join(path2data, f))
    h = im.shape[0]
    w = im.shape[1]
    print(f"old:{w},{h}")
    w = np.int16(np.round(w*resize))
    h = np.int16(np.round(h*resize))
    print(f"new:{w},{h}")
    image_resized = cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    for i in range(0,np.int16(size/2)):
        for j in range(0,w,np.int16(size/2)):
            patch = image_resized[i:i+size, j:j+size]
            if patch.shape[0] == patch.shape[1] and patch.shape[0] == size:
                k += 1
                imsave(os.path.join(outpath, f"{k}_{f}"), patch)