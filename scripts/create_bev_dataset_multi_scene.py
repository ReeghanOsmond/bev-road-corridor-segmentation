from pathlib import Path
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego, read_lidar_sweep


DATA_ROOT = Path("data/av2/sensor/val")
OUTPUT_ROOT = Path("outputs/dataset")

INPUT_DIR = OUTPUT_ROOT / "inputs"
DRIVABLE_LABEL_DIR = OUTPUT_ROOT / "labels_drivable"
LANE_LABEL_DIR = OUTPUT_ROOT / "labels_lane"

CLEAN_OUTPUT = True

MAX_SCENES = 10
MAX_SAMPLES_PER_SCENE = 50
SAMPLE_STRIDE = 5

X_MIN = -10.0
X_MAX = 60.0
LATERAL_MIN = -25.0
LATERAL_MAX = 25.0
Z_MIN = -3.0
Z_MAX = 5.0
RESOLUTION_M = 0.25

HEIGHT_PX = int((X_MAX - X_MIN) / RESOLUTION_M)
WIDTH_PX = int((LATERAL_MAX - LATERAL_MIN) / RESOLUTION_M)


def reset_output_dirs() -> None:
    if CLEAN_OUTPUT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DRIVABLE_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    LANE_LABEL_DIR.mkdir(parents=True, exist_ok=True)


def ego_to_bev_pixels(x_ego: np.ndarray, y_ego: np.ndarray) -> np.ndarray:
    lateral_display = -y_ego

    u = (lateral_display - LATERAL_MIN) / RESOLUTION_M
    v = (X_MAX - x_ego) / RESOLUTION_M

    pixels = np.column_stack([u, v])
    return np.round(pixels).astype(np.int32)


def create_lidar_bev_input(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    crop_mask = (
        (x > X_MIN) & (x < X_MAX) &
        (-y > LATERAL_MIN) & (-y < LATERAL_MAX) &
        (z > Z_MIN) & (z < Z_MAX)
    )

    x = x[crop_mask]
    y = y[crop_mask]
    z = z[crop_mask]

    lateral_display = -y

    u = np.round((lateral_display - LATERAL_MIN) / RESOLUTION_M).astype(np.int32)
    v = np.round((X_MAX - x) / RESOLUTION_M).astype(np.int32)

    valid_pixel_mask = (
        (u >= 0) & (u < WIDTH_PX) &
        (v >= 0) & (v < HEIGHT_PX)
    )

    u = u[valid_pixel_mask]
    v = v[valid_pixel_mask]
    z = z[valid_pixel_mask]

    density = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    height_sum = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    height_count = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    max_height = np.full((HEIGHT_PX, WIDTH_PX), fill_value=Z_MIN, dtype=np.float32)

    np.add.at(density, (v, u), 1.0)
    np.add.at(height_sum, (v, u), z)
    np.add.at(height_count, (v, u), 1.0)
    np.maximum.at(max_height, (v, u), z)

    mean_height = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    nonzero_mask = height_count > 0
    mean_height[nonzero_mask] = height_sum[nonzero_mask] / height_count[nonzero_mask]

    density_norm = np.log1p(density)

    if density_norm.max() > 0:
        density_norm = density_norm / density_norm.max()

    max_height_norm = (max_height - Z_MIN) / (Z_MAX - Z_MIN)
    max_height_norm = np.clip(max_height_norm, 0.0, 1.0)

    mean_height_norm = (mean_height - Z_MIN) / (Z_MAX - Z_MIN)
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

    return bev_input


def create_bev_labels(static_map: ArgoverseStaticMap, ego_SE3_city) -> tuple[np.ndarray, np.ndarray]:
    drivable_area_mask = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.uint8)
    lane_boundary_mask = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.uint8)

    for drivable_area in static_map.vector_drivable_areas.values():
        xyz_city = drivable_area.xyz
        xyz_ego = ego_SE3_city.transform_point_cloud(xyz_city)

        x_area = xyz_ego[:, 0]
        y_area = xyz_ego[:, 1]

        if (
            np.max(x_area) < X_MIN or np.min(x_area) > X_MAX or
            np.max(-y_area) < LATERAL_MIN or np.min(-y_area) > LATERAL_MAX
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
                np.max(x_lane) < X_MIN or np.min(x_lane) > X_MAX or
                np.max(-y_lane) < LATERAL_MIN or np.min(-y_lane) > LATERAL_MAX
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

    return drivable_area_mask, lane_boundary_mask


def main() -> None:
    reset_output_dirs()

    scenes = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[:MAX_SCENES]

    metadata = []
    global_sample_index = 0

    print(f"Scenes selected: {len(scenes)}")
    print(f"Maximum samples per scene: {MAX_SAMPLES_PER_SCENE}")
    print(f"Sample stride: {SAMPLE_STRIDE}")
    print(f"BEV input shape: 3 by {HEIGHT_PX} by {WIDTH_PX}")

    for scene_index, scene in enumerate(scenes):
        print(f"\nProcessing scene {scene_index + 1}/{len(scenes)}: {scene.name}")

        lidar_dir = scene / "sensors" / "lidar"
        map_dir = scene / "map"

        lidar_files = sorted(lidar_dir.glob("*.feather"))
        lidar_files = lidar_files[::SAMPLE_STRIDE][:MAX_SAMPLES_PER_SCENE]

        city_SE3_ego_by_timestamp = read_city_SE3_ego(scene)

        map_json_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        static_map = ArgoverseStaticMap.from_json(map_json_path)

        for lidar_file in tqdm(lidar_files, desc="Creating samples"):
            lidar_timestamp_ns = int(lidar_file.stem)

            if lidar_timestamp_ns not in city_SE3_ego_by_timestamp:
                continue

            points = read_lidar_sweep(lidar_file, attrib_spec="xyz")

            city_SE3_ego = city_SE3_ego_by_timestamp[lidar_timestamp_ns]
            ego_SE3_city = city_SE3_ego.inverse()

            bev_input = create_lidar_bev_input(points)
            drivable_area_mask, lane_boundary_mask = create_bev_labels(
                static_map=static_map,
                ego_SE3_city=ego_SE3_city,
            )

            sample_name = f"sample_{global_sample_index:06d}"

            input_path = INPUT_DIR / f"{sample_name}.npy"
            drivable_label_path = DRIVABLE_LABEL_DIR / f"{sample_name}.png"
            lane_label_path = LANE_LABEL_DIR / f"{sample_name}.png"

            np.save(input_path, bev_input)
            cv2.imwrite(str(drivable_label_path), drivable_area_mask)
            cv2.imwrite(str(lane_label_path), lane_boundary_mask)

            metadata.append(
                {
                    "sample_name": sample_name,
                    "scene_index": scene_index,
                    "scene": scene.name,
                    "lidar_timestamp_ns": lidar_timestamp_ns,
                    "lidar_file": str(lidar_file),
                    "input_path": str(input_path),
                    "drivable_label_path": str(drivable_label_path),
                    "lane_label_path": str(lane_label_path),
                    "x_min_m": X_MIN,
                    "x_max_m": X_MAX,
                    "lateral_min_m": LATERAL_MIN,
                    "lateral_max_m": LATERAL_MAX,
                    "resolution_m": RESOLUTION_M,
                }
            )

            global_sample_index += 1

    metadata_path = OUTPUT_ROOT / "metadata.json"

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Created samples: {len(metadata)}")
    print(f"Inputs saved to: {INPUT_DIR}")
    print(f"Drivable labels saved to: {DRIVABLE_LABEL_DIR}")
    print(f"Lane labels saved to: {LANE_LABEL_DIR}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()