from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from av2.utils.io import read_lidar_sweep


DATA_ROOT = Path("data/av2/sensor/val")
BEV_INPUT_DIR = Path("outputs/bev_inputs")
FIGURE_DIR = Path("outputs/figures")

BEV_INPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

scene = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[0]

camera_dir = scene / "sensors" / "cameras" / "ring_front_center"
lidar_dir = scene / "sensors" / "lidar"

camera_file = sorted(camera_dir.glob("*.jpg"))[0]
camera_timestamp_ns = int(camera_file.stem)

lidar_files = sorted(lidar_dir.glob("*.feather"))
lidar_file = min(
    lidar_files,
    key=lambda path: abs(int(path.stem) - camera_timestamp_ns),
)
lidar_timestamp_ns = int(lidar_file.stem)

time_difference_ms = abs(camera_timestamp_ns - lidar_timestamp_ns) / 1_000_000

print(f"Scene: {scene.name}")
print(f"Camera file: {camera_file.name}")
print(f"LiDAR file: {lidar_file.name}")
print(f"Time difference: {time_difference_ms:.2f} ms")

points = read_lidar_sweep(lidar_file, attrib_spec="xyz")

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

x_min = -10.0
x_max = 60.0
lateral_min = -25.0
lateral_max = 25.0
z_min = -3.0
z_max = 5.0
resolution_m = 0.25

height_px = int((x_max - x_min) / resolution_m)
width_px = int((lateral_max - lateral_min) / resolution_m)

crop_mask = (
    (x > x_min) & (x < x_max) &
    (y > -lateral_max) & (y < -lateral_min) &
    (z > z_min) & (z < z_max)
)

x = x[crop_mask]
y = y[crop_mask]
z = z[crop_mask]

lateral_display = -y

u = np.round((lateral_display - lateral_min) / resolution_m).astype(np.int32)
v = np.round((x_max - x) / resolution_m).astype(np.int32)

valid_pixel_mask = (
    (u >= 0) & (u < width_px) &
    (v >= 0) & (v < height_px)
)

u = u[valid_pixel_mask]
v = v[valid_pixel_mask]
z = z[valid_pixel_mask]

density = np.zeros((height_px, width_px), dtype=np.float32)
height_sum = np.zeros((height_px, width_px), dtype=np.float32)
height_count = np.zeros((height_px, width_px), dtype=np.float32)
max_height = np.full((height_px, width_px), fill_value=z_min, dtype=np.float32)

np.add.at(density, (v, u), 1.0)
np.add.at(height_sum, (v, u), z)
np.add.at(height_count, (v, u), 1.0)
np.maximum.at(max_height, (v, u), z)

mean_height = np.zeros((height_px, width_px), dtype=np.float32)
nonzero_mask = height_count > 0
mean_height[nonzero_mask] = height_sum[nonzero_mask] / height_count[nonzero_mask]

density_norm = np.log1p(density)
density_norm = density_norm / density_norm.max()

max_height_norm = (max_height - z_min) / (z_max - z_min)
max_height_norm = np.clip(max_height_norm, 0.0, 1.0)

mean_height_norm = (mean_height - z_min) / (z_max - z_min)
mean_height_norm = np.clip(mean_height_norm, 0.0, 1.0)
mean_height_norm[~nonzero_mask] = 0.0

bev_input = np.stack(
    [
        density_norm,
        max_height_norm,
        mean_height_norm,
    ],
    axis=0,
).astype(np.float32)

bev_input_uint8 = np.stack(
    [
        (density_norm * 255).astype(np.uint8),
        (max_height_norm * 255).astype(np.uint8),
        (mean_height_norm * 255).astype(np.uint8),
    ],
    axis=2,
)

npy_path = BEV_INPUT_DIR / "lidar_bev_input.npy"
png_path = BEV_INPUT_DIR / "lidar_bev_input.png"
figure_path = FIGURE_DIR / "lidar_bev_input_channels.png"

np.save(npy_path, bev_input)
cv2.imwrite(str(png_path), cv2.cvtColor(bev_input_uint8, cv2.COLOR_RGB2BGR))

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

axes[0].imshow(density_norm, cmap="viridis")
axes[0].set_title("Point density")
axes[0].axis("off")

axes[1].imshow(max_height_norm, cmap="viridis")
axes[1].set_title("Max height")
axes[1].axis("off")

axes[2].imshow(mean_height_norm, cmap="viridis")
axes[2].set_title("Mean height")
axes[2].axis("off")

plt.suptitle("LiDAR BEV input channels")
plt.tight_layout()
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved raw BEV input to: {npy_path}")
print(f"Saved RGB BEV input image to: {png_path}")
print(f"Saved channel figure to: {figure_path}")
print(f"BEV input shape: {bev_input.shape}")
print(f"Number of LiDAR points used: {len(z)}")