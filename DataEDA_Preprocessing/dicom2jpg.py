import os
import numpy as np
import cv2
import pydicom
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from functools import partial
from sklearn.model_selection import train_test_split

WORK_DIR = "/home/minhyekj/data/dataset/physionet.org/files/vindr-cxr"

dicom_files = os.listdir(os.path.join(WORK_DIR, "1.0.0", "dicom", "test"))
dicom_files = [f for f in dicom_files if f.endswith(".dicom")]
print(len(dicom_files))


def resize_and_save(load_path, save_path):  # load_path=/path/to/load/*.dicom, save_path=/path/to/save/*.jpg
    ds = pydicom.dcmread(load_path, force=True)
    img = ds.pixel_array
    img = apply_modality_lut(img, ds)  # rescaleSlope & intercept
    img = apply_voi_lut(img, ds)  # windowing
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation.lower().strip() == "monochrome1":
            img = img.max() - img  # invert
    
    h, w = img.shape
    ratio = 512 / min(h, w)
    target_size = (int(w * ratio), int(h * ratio))
    img = cv2.resize(img, target_size, cv2.INTER_LANCZOS4)
   
    # normalize
    img = (img - img.min()) / (img.max() - img.min()) * np.iinfo(np.uint8).max
    img = img.astype(np.uint8)
    cv2.imwrite(save_path, img)


def resize_and_save_wrapper(dicom_file):
    resize_and_save(os.path.join(WORK_DIR, "1.0.0", "dicom", "test", dicom_file),
                    os.path.join(WORK_DIR, "1.0.0", "jpg", "test", dicom_file.replace(".dicom", ".jpg")))

# Use multiprocessing to parallelize the resizing and saving process
with Pool(48) as pool:
    pool.map(resize_and_save_wrapper, dicom_files)