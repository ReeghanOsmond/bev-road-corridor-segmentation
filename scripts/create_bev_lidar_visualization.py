from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from av2.utils.io import read_lidar_sweep


DATA_ROOT = Path("data/av2/sensor/val")
CAMERA_NAME = "ring_front_center"
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

scene = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[0]

camera_dir = scene / "sensors" / "cameras" / CAMERA_NAME
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

# Crop to a useful BEV region around the ego vehicle.
mask = (
    (x > -10.0) & (x < 60.0) &
    (y > -25.0) & (y < 25.0) &
    (z > -3.0) & (z < 5.0)
)

x = x[mask]
y = y[mask]
z = z[mask]

plt.figure(figsize=(8, 10))
scatter = plt.scatter(
    -y,
    x,
    c=z,
    s=0.3,
    cmap="viridis",
)

plt.scatter(
    0,
    0,
    c="red",
    s=50,
    marker="x",
    label="ego vehicle",
)

plt.xlabel("left / right position (m)")
plt.ylabel("forward position (m)")
plt.title("BEV LiDAR visualization")
plt.xlim(-25, 25)
plt.ylim(-10, 60)
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()
plt.colorbar(scatter, label="height z (m)")

output_path = OUTPUT_DIR / "bev_lidar_visualization.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")
print(f"Number of plotted points: {len(x)}")