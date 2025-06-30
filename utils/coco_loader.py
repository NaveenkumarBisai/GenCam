# coco_loader.py

import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Paths
IMAGES_DIR = "data/coco/val2017"  # Your subset folder
ANNOTATIONS_FILE = "data/coco/annotations/captions_val2017.json"

# Load annotations
with open(ANNOTATIONS_FILE, 'r') as f:
    captions_data = json.load(f)

# Build image_id to filename map
id_to_filename = {img['id']: img['file_name'] for img in captions_data['images']}

# Build image_id to captions map
from collections import defaultdict
image_captions = defaultdict(list)
for ann in captions_data['annotations']:
    image_captions[ann['image_id']].append(ann['caption'])

# Filter to only those in your local subset
available_files = set(os.listdir(IMAGES_DIR))
valid_ids = [img_id for img_id, fname in id_to_filename.items() if fname in available_files]

# Random preview
random.shuffle(valid_ids)
for img_id in valid_ids[:5]:
    fname = id_to_filename[img_id]
    img_path = os.path.join(IMAGES_DIR, fname)
    captions = image_captions[img_id]

    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('\n'.join(captions[:2]))  # Show top 2 captions
    plt.show()
