# EN3160 — Image Processing and Computer Vision

Coursework for **EN3160 Image Processing and Computer Vision**, Department of Electronic & Telecommunication Engineering, University of Moratuwa (2025).
Implemented in Python with OpenCV, NumPy and scikit-image.

**Author:** Kumarage R. V. (220343B)

## Contents

| Folder / file | Topic |
|---|---|
| [`Assignment 01`](Assignment%2001) | Intensity transformations and neighbourhood filtering |
| [`Assignment 02`](Assignment%2002) | Blob detection, RANSAC fitting, homography and image stitching |
| [`Research Paper`](Research%20Paper) | Paper selection and reproduction study of Zero-Shot Composed Image Retrieval (ZS-CIR) |
| [`a_image_basics.ipynb`](a_image_basics.ipynb), [`b_spatial_filtering.ipynb`](b_spatial_filtering.ipynb) | Tutorial notebooks: image basics, spatial filtering |

## Assignment 01 — Intensity transformations and filtering

Notebook: [`Assignment 01/Notebooks/Assignment1.ipynb`](Assignment%2001/Notebooks/Assignment1.ipynb), helpers in [`src/utils.py`](Assignment%2001/src/utils.py).

1. Piecewise intensity transformation from a given curve
2. Accentuating white and gray matter in a brain proton-density slice
3. Gamma correction on the L\* channel (Lab) with histograms
4. Vibrance enhancement on the S channel (HSV)
5. Histogram equalisation written from scratch
6. Foreground-only histogram equalisation with an HSV mask
7. Sobel filtering three ways: `filter2D`, manual convolution and separable kernels
8. Zooming with nearest-neighbour and bilinear interpolation, scored with SSD
9. GrabCut segmentation with a blurred background

| Sobel comparison | GrabCut + blurred background |
|---|---|
| ![Sobel](Assignment%2001/figures/task7_cmp.png) | ![GrabCut](Assignment%2001/figures/task9_cmp.png) |

## Assignment 02 — Features, fitting and geometry

| Notebook | Task |
|---|---|
| [`Q1`](Assignment%2002/Assignment02_Q1.ipynb) | Detect sunflowers with Laplacian-of-Gaussian blob detection (`blob_log`) and a feathered horizon mask |
| [`Q2`](Assignment%2002/Assignment02_Q2.ipynb) | RANSAC to fit a line and a circle to noisy points, and why fitting the circle first is less robust |
| [`Q3`](Assignment%2002/Assignment02_Q3.ipynb) | Homography warping to superimpose an image onto a planar surface, with a feathered mask and normal, multiply or overlay blending |
| [`Q4`](Assignment%2002/Assignment02_Q4.ipynb) | SIFT matching, homography from RANSAC, and stitching `img1` onto `img5` of the Oxford Graffiti set |

| RANSAC line + circle | Homography overlay | SIFT stitching |
|---|---|---|
| ![RANSAC](Assignment%2002/q2_results.png) | ![Overlay](Assignment%2002/overlay.png) | ![Stitched](Assignment%2002/stitched.png) |

## Research paper

Reproduction and extension study of **ZS-CIR** (Zero-Shot Composed Image Retrieval), based on [Code-kunkun/ZS-CIR](https://github.com/Code-kunkun/ZS-CIR).
The folder holds the candidate papers, the repository comparison used to choose one, and the reproducibility assessment.
The implementation lives in [ZS-CIR-EN3160-project](https://github.com/rivindu02/ZS-CIR-EN3160-project).

## Run

```bash
pip install opencv-python numpy matplotlib scikit-image jupyter
jupyter notebook
```

Some Assignment 02 notebooks were written in Google Colab and read images from a Drive `BASE_PATH`. Change that path to the local `Assignment 02/` folder.
