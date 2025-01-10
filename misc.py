import os
import json
from pycocotools.coco import COCO
import requests
from io import BytesIO
from PIL import Image

# --- Configuration ---
annFile = 'annotations/instances_val2017.json'  # Annotation file path
output_dir = 'val2017'  # Output directory for images

# --- Download annotations if not present ---
if not os.path.exists(annFile):
 os.makedirs(os.path.dirname(annFile), exist_ok=True)
 annotation_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
 print(f"Downloading annotations from {annotation_url}")
 r = requests.get(annotation_url)
 import zipfile
 with zipfile.ZipFile(BytesIO(r.content)) as zip_ref:
     zip_ref.extractall('.')

# --- Create output directory ---
os.makedirs(output_dir, exist_ok=True)

# --- Initialize COCO API ---
coco = COCO(annFile)

# --- Get image IDs for the validation set ---
imgIds = coco.getImgIds(catIds=[])  # Get all image IDs

# --- Download images ---
for imgId in imgIds:
 img_info = coco.loadImgs([imgId])[0]
 img_url = img_info['coco_url']
 file_name = img_info['file_name']
 output_path = os.path.join(output_dir, file_name)

 if not os.path.exists(output_path):  # Check if the image already exists
     try:
         print(f"Downloading {img_url} to {output_path}")
         r = requests.get(img_url, stream=True)
         r.raise_for_status()  # Raise an exception for bad status codes (e.g., 404)
         image = Image.open(BytesIO(r.content))
         image.save(output_path)
     except requests.exceptions.RequestException as e:
         print(f"Error downloading {img_url}: {e}")
 else:
     print(f"Skipping {file_name} (already exists)")

print("Finished downloading val2017 images.")