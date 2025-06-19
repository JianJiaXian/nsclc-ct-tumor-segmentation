import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 設定 input 與 output 路徑
input_dir = "./npy_files/inference_results_paper"
output_dir = "./nifti_files/inference_results_paper"
os.makedirs(output_dir, exist_ok=True)

# 批次處理所有 npy
for file in tqdm(sorted(os.listdir(input_dir))):
    if not file.endswith(".npy"):
        continue

    patient_id = file.replace(".npy", "")
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, f"{patient_id}.nii.gz")

    # 讀取 npy
    pred_array = np.load(input_path)
    
    # 注意 SimpleITK 需要 ZYX 排列 (npy 預設是 Z,Y,X 沒問題)
    pred_sitk = sitk.GetImageFromArray(pred_array.astype(np.uint8))
    
    # 可以選擇加上 isotropic spacing (假設前面你 resample 過)
    pred_sitk.SetSpacing((1.0, 1.0, 1.0))  

    # 輸出 NIfTI
    sitk.WriteImage(pred_sitk, output_path)

    print(f"{patient_id} 轉換完成")

print("== 全部轉檔完成 ==")
