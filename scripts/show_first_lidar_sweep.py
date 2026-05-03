from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_ROOT = Path("data/av2/sensor/val")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

scene = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[0]
lidar_dir = scene / "sensors" / "lidar"

lidar_file = sorted(lidar_dir.glob("*.feather"))[0]

print(f"Scene: {scene.name}")
print(f"LiDAR file: {lidar_file.name}")

points = pd.read_feather(lidar_file)

print(points.head())
print(points.columns)

x = points["x"]
y = points["y"]

plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=0.2)
plt.axis("equal")
plt.xlabel("x position")
plt.ylabel("y position")
plt.title("First Argoverse LiDAR sweep in top down view")

output_path = OUTPUT_DIR / "first_lidar_sweep_bev.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")