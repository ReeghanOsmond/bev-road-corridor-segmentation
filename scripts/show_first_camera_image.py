from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


DATA_ROOT = Path("data/av2/sensor/val")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Data folder not found: {DATA_ROOT}")

front_center_images = sorted(DATA_ROOT.rglob("sensors/cameras/ring_front_center/*.jpg"))

image_path = front_center_images[0]

print(f"Using image:")
print(image_path)

image = Image.open(image_path).convert("RGB")

output_path = OUTPUT_DIR / "first_camera_image.png"
image.save(output_path)

print(f"Saved image to:")
print(output_path)

plt.figure(figsize=(12, 7))
plt.imshow(image)
plt.axis("off")
plt.title("First Argoverse camera image")
plt.show()