from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.utils.io import read_city_SE3_ego, read_img, read_lidar_sweep


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

image = read_img(camera_file, channel_order="RGB")
points_lidar_time = read_lidar_sweep(lidar_file, attrib_spec="xyz")

camera = PinholeCamera.from_feather(
    log_dir=scene,
    cam_name=CAMERA_NAME,
)

city_SE3_ego_by_timestamp = read_city_SE3_ego(scene)

city_SE3_ego_cam_t = city_SE3_ego_by_timestamp[camera_timestamp_ns]
city_SE3_ego_lidar_t = city_SE3_ego_by_timestamp[lidar_timestamp_ns]

uv, points_cam, is_valid = camera.project_ego_to_img_motion_compensated(
    points_lidar_time=points_lidar_time,
    city_SE3_ego_cam_t=city_SE3_ego_cam_t,
    city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
)

uv_valid = uv[is_valid]
points_cam_valid = points_cam[is_valid]

depth_m = points_cam_valid[:, 2]

plt.figure(figsize=(14, 8))
plt.imshow(image)
scatter = plt.scatter(
    uv_valid[:, 0],
    uv_valid[:, 1],
    c=depth_m,
    s=0.4,
    cmap="turbo",
)
plt.colorbar(scatter, label="Depth in camera frame, meters")
plt.axis("off")
plt.title("LiDAR projected onto front camera image")

output_path = OUTPUT_DIR / "lidar_projected_to_front_camera.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Valid projected points: {len(uv_valid)}")
print(f"Saved figure to: {output_path}")