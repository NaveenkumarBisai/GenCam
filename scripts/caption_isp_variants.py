import os
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Paths
variants_dir = "results/"
image_variants = ["original.jpg", "dark.jpg", "bright.jpg", "low_contrast.jpg", "sharper.jpg"]
image_path_map = {v: os.path.join(variants_dir, v) for v in image_variants}

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Generate captions for each variant
records = []
for variant_name, path in image_path_map.items():
    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    records.append({
        "variant": variant_name,
        "caption": caption
    })

# Save results
df = pd.DataFrame(records)
df.to_csv("results/captions/isp_variant_captions.csv", index=False)
print("✅ Captions saved to results/captions/isp_variant_captions.csv")
