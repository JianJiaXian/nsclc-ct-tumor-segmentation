import os
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes, label
import matplotlib.pyplot as plt

# Params
TARGET_SPACING = (1.0, 1.0, 1.0)
WINDOW_WW = 1500
WINDOW_WL = -600
FINAL_SIZE = 512
DILATION_ITER = 5

def load_dicom_to_itk(dicom_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def resample_image(itk_image, out_spacing=TARGET_SPACING):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_size = [
        int(round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkLinear)
    
    return resample.Execute(itk_image)

def normalize_to_uint8(image_array):
    img = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    img = img * 255.0
    return img.astype(np.uint8)

def crop_and_pad(slice_array, target_size=FINAL_SIZE):
    h, w = slice_array.shape
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2
    padded = np.pad(slice_array, ((pad_h, target_size - h - pad_h),
                                   (pad_w, target_size - w - pad_w)),
                    mode='constant', constant_values=0)
    return padded

#---Lung isolation part
def lung_isolation(resampled_itk, dilation_iter=5):
    inferer = LMInferer('R231')
    raw_mask = inferer.apply(resampled_itk)

    # 先進行 dilation + closing（粗遮罩）
    lung_mask = binary_dilation(raw_mask, iterations=dilation_iter)
    lung_mask = binary_closing(lung_mask, structure=np.ones((3, 3, 3)))

    # Connected components → 只保留兩個最大連通區塊（左右肺）
    labeled, num = label(lung_mask)
    component_sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    top2 = np.argsort(component_sizes)[-2:] + 1
    lung_mask_filtered = np.isin(labeled, top2)

    # 對每個 axial slice 進行 hole filling
    for i in range(lung_mask_filtered.shape[0]):
        lung_mask_filtered[i] = binary_fill_holes(lung_mask_filtered[i])

    return lung_mask_filtered.astype(np.uint8)
#---

# others 

def visualize_mask_overlay(image_slice, mask_slice):
    plt.imshow(image_slice, cmap='gray')
    plt.contour(mask_slice, colors='r', linewidths=0.5)
    plt.title('Lung Mask Overlay')
    plt.axis('off')
    plt.show()

# per patients

def preprocess_single_patient(dicom_path, visualize=False):
    itk_image = load_dicom_to_itk(dicom_path)
    resampled_itk = resample_image(itk_image)
    resampled_array = sitk.GetArrayFromImage(resampled_itk)

    lung_mask_array = lung_isolation(resampled_itk)
    masked_array = np.where(lung_mask_array > 0, resampled_array, resampled_array.min())

    processed_slices = []
    for i in range(masked_array.shape[0]):
        slice_img = masked_array[i]
        slice_img = np.clip(slice_img, -1350, 150)
        slice_img = normalize_to_uint8(slice_img)
        slice_img = crop_and_pad(slice_img)

        if visualize and i == masked_array.shape[0] // 2:
            visualize_mask_overlay(slice_img, lung_mask_array[i])

        processed_slices.append(slice_img)

    processed_array = np.stack(processed_slices, axis=0)
    return processed_array

# batch of patients

def batch_preprocess(dicom_root_dir, save_dir):
    dicom_dirs = []

    for patient_folder in os.listdir(dicom_root_dir):
        patient_path = os.path.join(dicom_root_dir, patient_folder)
        if os.path.isdir(patient_path):
            for subfolder in os.listdir(patient_path):
                subfolder_path = os.path.join(patient_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for series_folder in os.listdir(subfolder_path):
                        if series_folder.startswith("NA-"):
                            dicom_dirs.append(os.path.join(subfolder_path, series_folder))

    os.makedirs(save_dir, exist_ok=True)

    for dicom_path in tqdm(dicom_dirs):
        patient_id = dicom_path.split(os.sep)[-3]
        output_path = os.path.join(save_dir, f"{patient_id}.npy")

        processed_array = preprocess_single_patient(dicom_path)
        np.save(output_path, processed_array)
        print(f"Saved: {output_path}")

# main

if __name__ == "__main__":
    dicom_root_dir = "/work/u5453836/NSCLC/manifest-1598890146597/NSCLC-Radiomics-Interobserver1"
    save_dir = "/work/u5453836/NSCLC/preprocessed_inference_ready_paper"
    batch_preprocess(dicom_root_dir, save_dir)
