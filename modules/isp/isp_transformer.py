from PIL import Image, ImageEnhance
import os

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def adjust_sharpness(image, factor):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

# Test on one image
if __name__ == "__main__":
    img_path = "data/coco/val2017/000000573626.jpg"
    image = Image.open(img_path).convert("RGB")

    image.save("results/original.jpg")
    adjust_brightness(image, 0.5).save("results/dark.jpg")
    adjust_brightness(image, 1.5).save("results/bright.jpg")
    adjust_contrast(image, 0.8).save("results/low_contrast.jpg")
    adjust_sharpness(image, 2.0).save("results/sharper.jpg")

    print("✅ ISP variants saved in /results")
