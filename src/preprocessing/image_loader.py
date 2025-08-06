
import os
from PIL import Image

def load_images_from_folder(folder_path, extensions={'.jpg', '.jpeg', '.png', '.bmp', '.gif'}):
    images = []
    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            try:
                img = Image.open(os.path.join(folder_path, filename))
                images.append((filename, img))
            except Exception as e:
                print(f"Error al cargar {filename}: {e}")
    return images
