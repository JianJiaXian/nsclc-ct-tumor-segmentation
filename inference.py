import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

import os
import numpy as np
from keras.models import model_from_json
from tqdm import tqdm

# 載入模型架構
with open('./weights/model_v7.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)

# 載入模型權重
model.load_weights('./weights/weights_v7.hdf5')

# Preprocessed 資料夾
input_dir = "./npy_files/preprocessed_inference_ready_paper"
output_dir = "./npy_files/inference_results_paper"
os.makedirs(output_dir, exist_ok=True)

# 批次 inference
for file in tqdm(sorted(os.listdir(input_dir))):
    if not file.endswith(".npy"):
        continue

    patient_id = file.replace(".npy", "")
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, f"{patient_id}_pred.npy")

    img_array = np.load(input_path)
    img_array = img_array.astype(np.float32) / 255.0

    pred_slices = []
    for i in range(img_array.shape[0]):
        slice_img = img_array[i]
        slice_img = np.expand_dims(slice_img, axis=0)  # (H, W) -> (1, H, W)
        slice_img = np.expand_dims(slice_img, axis=-1) # (1, H, W) -> (1, H, W, 1)
        pred = model.predict(slice_img)
        pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8)
        pred_slices.append(pred_mask)

    pred_volume = np.stack(pred_slices, axis=0)
    np.save(output_path, pred_volume)

print("All inference done!")
