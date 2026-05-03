from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego


DATA_ROOT = Path("data/av2/sensor/val")
BEV_LABEL_DIR = Path("outputs/bev_labels")
FIGURE_DIR = Path("outputs/figures")

BEV_LABEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

scene = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[0]

camera_dir = scene / "sensors" / "cameras" / "ring_front_center"
lidar_dir = scene / "sensors" / "lidar"
map_dir = scene / "map"

camera_file = sorted(camera_dir.glob("*.jpg"))[0]
camera_timestamp_ns = int(camera_file.stem)

lidar_files = sorted(lidar_dir.glob("*.feather"))
lidar_file = min(
    lidar_files,
    key=lambda path: abs(int(path.stem) - camera_timestamp_ns),
)
lidar_timestamp_ns = int(lidar_file.stem)

city_SE3_ego_by_timestamp = read_city_SE3_ego(scene)
city_SE3_ego = city_SE3_ego_by_timestamp[lidar_timestamp_ns]
ego_SE3_city = city_SE3_ego.inverse()

map_json_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
static_map = ArgoverseStaticMap.from_json(map_json_path)

x_min = -10.0
x_max = 60.0
lateral_min = -25.0
lateral_max = 25.0
resolution_m = 0.25

height_px = int((x_max - x_min) / resolution_m)
width_px = int((lateral_max - lateral_min) / resolution_m)

drivable_area_mask = np.zeros((height_px, width_px), dtype=np.uint8)
lane_boundary_mask = np.zeros((height_px, width_px), dtype=np.uint8)

# Okabe-Ito style colorblind-friendly colors in RGB
DRIVABLE_AREA_COLOR = np.array([86, 180, 233], dtype=np.uint8)   # sky blue
LANE_BOUNDARY_COLOR = np.array([213, 94, 0], dtype=np.uint8)     # vermillion
BACKGROUND_COLOR = np.array([255, 255, 255], dtype=np.uint8)     # white


def ego_to_bev_pixels(x_ego: np.ndarray, y_ego: np.ndarray) -> np.ndarray:
    """
    Convert ego-frame coordinates into BEV image pixel coordinates.

    Ego frame:
        x = forward
        y = left

    Display convention:
        screen left = vehicle left
        screen right = vehicle right
        screen up = forward
    """
    lateral_display = -y_ego

    u = (lateral_display - lateral_min) / resolution_m
    v = (x_max - x_ego) / resolution_m

    pixels = np.column_stack([u, v])
    return np.round(pixels).astype(np.int32)


for drivable_area in static_map.vector_drivable_areas.values():
    xyz_city = drivable_area.xyz
    xyz_ego = ego_SE3_city.transform_point_cloud(xyz_city)

    x_area = xyz_ego[:, 0]
    y_area = xyz_ego[:, 1]

    if (
        np.max(x_area) < x_min or np.min(x_area) > x_max or
        np.max(-y_area) < lateral_min or np.min(-y_area) > lateral_max
    ):
        continue

    polygon_pixels = ego_to_bev_pixels(x_area, y_area)

    cv2.fillPoly(
        drivable_area_mask,
        [polygon_pixels],
        color=255,
    )


for lane_segment in static_map.vector_lane_segments.values():
    for lane_boundary in [
        lane_segment.left_lane_boundary,
        lane_segment.right_lane_boundary,
    ]:
        xyz_city = lane_boundary.xyz
        xyz_ego = ego_SE3_city.transform_point_cloud(xyz_city)

        x_lane = xyz_ego[:, 0]
        y_lane = xyz_ego[:, 1]

        if (
            np.max(x_lane) < x_min or np.min(x_lane) > x_max or
            np.max(-y_lane) < lateral_min or np.min(-y_lane) > lateral_max
        ):
            continue

        line_pixels = ego_to_bev_pixels(x_lane, y_lane)

        cv2.polylines(
            lane_boundary_mask,
            [line_pixels],
            isClosed=False,
            color=255,
            thickness=2,
        )


drivable_area_mask_path = BEV_LABEL_DIR / "drivable_area_mask.png"
lane_boundary_mask_path = BEV_LABEL_DIR / "lane_boundary_mask.png"
combined_bev_labels_path = BEV_LABEL_DIR / "combined_bev_labels.png"
annotated_figure_path = FIGURE_DIR / "bev_labels_with_legend.png"

cv2.imwrite(str(drivable_area_mask_path), drivable_area_mask)
cv2.imwrite(str(lane_boundary_mask_path), lane_boundary_mask)

combined_rgb = np.full((height_px, width_px, 3), BACKGROUND_COLOR, dtype=np.uint8)
combined_rgb[drivable_area_mask > 0] = DRIVABLE_AREA_COLOR
combined_rgb[lane_boundary_mask > 0] = LANE_BOUNDARY_COLOR

# Save the color combined label image in outputs/bev_labels
combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(combined_bev_labels_path), combined_bgr)

plt.figure(figsize=(8, 10))
plt.imshow(
    combined_rgb,
    extent=[lateral_min, lateral_max, x_min, x_max],
    origin="upper",
)
plt.xlabel("lateral position (m)")
plt.ylabel("forward position (m)")
plt.title("BEV labels")
plt.xlim(lateral_min, lateral_max)
plt.ylim(x_min, x_max)

ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

legend_handles = [
    Patch(
        facecolor=DRIVABLE_AREA_COLOR / 255.0,
        edgecolor="none",
        label="Drivable area",
    ),
    Patch(
        facecolor=LANE_BOUNDARY_COLOR / 255.0,
        edgecolor="none",
        label="Lane boundaries",
    ),
]

plt.legend(handles=legend_handles, loc="upper right")
plt.grid(True, alpha=0.2)
plt.savefig(annotated_figure_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Scene: {scene.name}")
print(f"Drivable area mask saved to: {drivable_area_mask_path}")
print(f"Lane boundary mask saved to: {lane_boundary_mask_path}")
print(f"Combined BEV labels saved to: {combined_bev_labels_path}")
print(f"Annotated figure saved to: {annotated_figure_path}")
print(f"Mask size: {height_px} by {width_px} pixels")
print(f"Resolution: {resolution_m} meters per pixel")