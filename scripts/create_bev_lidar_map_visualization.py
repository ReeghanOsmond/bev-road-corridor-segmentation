from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego, read_lidar_sweep


DATA_ROOT = Path("data/av2/sensor/val")
CAMERA_NAME = "ring_front_center"
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

scene = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[0]

camera_dir = scene / "sensors" / "cameras" / CAMERA_NAME
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

time_difference_ms = abs(camera_timestamp_ns - lidar_timestamp_ns) / 1_000_000

print(f"Scene: {scene.name}")
print(f"Camera file: {camera_file.name}")
print(f"LiDAR file: {lidar_file.name}")
print(f"Time difference: {time_difference_ms:.2f} ms")

points = read_lidar_sweep(lidar_file, attrib_spec="xyz")

city_SE3_ego_by_timestamp = read_city_SE3_ego(scene)
city_SE3_ego = city_SE3_ego_by_timestamp[lidar_timestamp_ns]
ego_SE3_city = city_SE3_ego.inverse()

map_json_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
static_map = ArgoverseStaticMap.from_json(map_json_path)

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

lidar_mask = (
    (x > -10.0) & (x < 60.0) &
    (y > -25.0) & (y < 25.0) &
    (z > -3.0) & (z < 5.0)
)

x = x[lidar_mask]
y = y[lidar_mask]
z = z[lidar_mask]

lidar_xy = np.column_stack([x, y])

plt.figure(figsize=(9, 10))

for drivable_area in static_map.vector_drivable_areas.values():
    xyz_city = drivable_area.xyz
    xyz_ego = ego_SE3_city.transform_point_cloud(xyz_city)

    x_area = xyz_ego[:, 0]
    y_area = xyz_ego[:, 1]

    if (
        np.max(x_area) < -10.0 or np.min(x_area) > 60.0 or
        np.max(y_area) < -25.0 or np.min(y_area) > 25.0
    ):
        continue

    polygon_xy = np.column_stack([x_area, y_area])
    polygon_path = MplPath(polygon_xy)

    lidar_inside_polygon = polygon_path.contains_points(lidar_xy)
    number_of_lidar_points_inside = np.count_nonzero(lidar_inside_polygon)

    if number_of_lidar_points_inside < 20:
        continue

    plt.fill(
        -y_area,
        x_area,
        facecolor="#D9D9D9",
        edgecolor="#4D4D4D",
        linewidth=1.0,
        alpha=0.45,
        zorder=1,
    )

scatter = plt.scatter(
    -y,
    x,
    c=z,
    s=0.25,
    cmap="viridis",
    alpha=0.6,
    zorder=2,
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
            np.max(x_lane) < -10.0 or np.min(x_lane) > 60.0 or
            np.max(y_lane) < -25.0 or np.min(y_lane) > 25.0
        ):
            continue

        plt.plot(
            -y_lane,
            x_lane,
            color="#CC79A7",
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )

plt.scatter(
    0,
    0,
    c="black",
    s=70,
    marker="x",
    zorder=4,
)

plt.xlabel("lateral position, vehicle left to screen left (m)")
plt.ylabel("forward position (m)")
plt.title("BEV LiDAR with map geometry")

plt.xlim(-25, 25)
plt.ylim(-10, 60)

ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

plt.grid(True, alpha=0.3)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#31688E",
        markeredgecolor="none",
        markersize=6,
        label="LiDAR points",
    ),
    Patch(
        facecolor="#D9D9D9",
        edgecolor="#4D4D4D",
        alpha=0.45,
        label="Drivable area",
    ),
    Line2D(
        [0],
        [0],
        color="#CC79A7",
        linewidth=2,
        label="Lane boundaries",
    ),
    Line2D(
        [0],
        [0],
        marker="x",
        color="black",
        linewidth=0,
        markersize=8,
        label="Ego vehicle",
    ),
]

plt.legend(handles=legend_handles, loc="upper right")
plt.colorbar(scatter, label="height z (m)")

output_path = OUTPUT_DIR / "bev_lidar_with_map_geometry.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")
print(f"Number of plotted LiDAR points: {len(x)}")