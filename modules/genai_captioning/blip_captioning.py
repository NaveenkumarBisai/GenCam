from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

# Generate caption
def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Test it
if __name__ == "__main__":
    test_image = "data/coco/val2017/000000573626.jpg"  # replace with an actual file
    caption = generate_caption(test_image)
    print("🧠 Caption:", caption)
