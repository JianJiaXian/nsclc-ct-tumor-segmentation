# NSCLC Lung Tumor Segmentation Pipeline

A full pipeline implementation for **automated detection and segmentation of non-small cell lung cancer (NSCLC)** in CT images, inspired by the paper:

> Primakov, S.P., Ibrahim, A., van Timmeren, J.E. et al.  
> *Automated detection and segmentation of non-small cell lung cancer computed tomography images*.  
> Nature Communications 13, 3423 (2022).  
>  [Paper Link](https://www.nature.com/articles/s41467-022-30841-3) |  [DOI](https://doi.org/10.1038/s41467-022-30841-3)

---

## Repository Contents

| File | Description |
|------|-------------|
| `preprocess_isolate_paper.py` | Preprocesses DICOM CT volumes and performs lung isolation using `lungmask` R231. |
| `inference.py` | Loads the pretrained model provided by the authors and performs slice-by-slice inference. |
| `npy2nifti.py` | Converts inference `.npy` results into `.nii.gz` format for 3D Slicer visualization. |
| `weights/model_v7.json` | Model architecture (from original paper). |
| `weights/weights_v7.hdf5` | Trained weights (from original paper). |

---

## Dataset

- **NSCLC-Radiomics-Interobserver1**  
  A public dataset from The Cancer Imaging Archive (TCIA).  
 [Download Here](https://www.cancerimagingarchive.net/collection/nsclc-radiomics-interobserver1/)

---

## Pipeline Overview

### Step 1: Preprocessing & Lung Isolation

```bash
python preprocess_isolate_paper.py
```

- Loads and resamples DICOM volumes to 1mm³ isotropic spacing.
- Applies lung isolation using the `lungmask` R231 model ([GitHub Repo](https://github.com/JoHof/lungmask)).
- Crops and normalizes each slice, output shape: `(Z, 512, 512)`
- Saves each patient’s preprocessed image as `.npy`.

### Step 2: Inference

```bash
python inference.py
```

- Loads the `model_v7` JSON + weights provided by the paper.
- Performs 2.5D slice-by-slice segmentation.
- Binarizes results (threshold 0.5).
- Saves predicted volume mask as `.npy`.

### Step 3: NIfTI Conversion for Visualization

```bash
python npy2nifti.py
```

- Converts `.npy` masks to `.nii.gz` format.
- Sets spacing to (1.0, 1.0, 1.0).
- Results can be visualized in [3D Slicer](https://www.slicer.org/).

---

## Sample Lung Isolation Method

Implemented in `preprocess_isolate_paper.py`:

```python
from lungmask import LMInferer
lung_mask = LMInferer('R231').apply(resampled_itk)

# Post-processing
lung_mask = binary_dilation(lung_mask, iterations=5)
lung_mask = binary_closing(lung_mask, structure=np.ones((3,3,3)))
...
```

> Keeps the largest two connected components (left/right lungs), followed by hole-filling.

---

## Model Information

- **Architecture**: 2.5D CNN (inference only)
- **Input shape**: `(1, 512, 512, 1)`
- **Output**: binary tumor mask per slice
- **Source**: model structure + weights from original authors

---

## Requirements

```bash
pip install numpy SimpleITK lungmask tqdm matplotlib scipy tensorflow keras
```

- TensorFlow (v1.x compatible, for inference script)
- Keras (<=2.2.4 recommended)

---

## Acknowledgements

- [Nature Communications Paper](https://doi.org/10.1038/s41467-022-30841-3)
- [TCIA Dataset](https://www.cancerimagingarchive.net/collection/nsclc-radiomics-interobserver1/)
- [lungmask by Hofmanninger et al.](https://github.com/JoHof/lungmask)
