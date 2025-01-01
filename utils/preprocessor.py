from PIL import Image


def load_image(image_file, transform=None):
    if isinstance(image_file, str):
        img = Image.open(image_file).convert('RGB')
    else:
        img = Image.open(image_file).convert('RGB')
    img = img.resize([224, 224], Image.LANCZOS)
    if transform is not None:
        img = transform(img).unsqueeze(0)
    return img
   