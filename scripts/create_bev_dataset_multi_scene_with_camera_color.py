from pathlib import Path
import bisect
import json
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego, read_img, read_lidar_sweep


DATA_ROOT = Path("data/av2/sensor/val")
OUTPUT_ROOT = Path("outputs/dataset_camera_color")

INPUT_DIR = OUTPUT_ROOT / "inputs"
DRIVABLE_LABEL_DIR = OUTPUT_ROOT / "labels_drivable"
LANE_LABEL_DIR = OUTPUT_ROOT / "labels_lane"

CLEAN_OUTPUT = True

MAX_SCENES = 10
MAX_SAMPLES_PER_SCENE = 50
SAMPLE_STRIDE = 5

RING_CAMERAS = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
]

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


def get_closest_file(files: list[Path], timestamps: list[int], target_timestamp_ns: int) -> Path:
    index = bisect.bisect_left(timestamps, target_timestamp_ns)

    if index == 0:
        return files[0]

    if index == len(timestamps):
        return files[-1]

    before = files[index - 1]
    after = files[index]

    before_difference = abs(int(before.stem) - target_timestamp_ns)
    after_difference = abs(int(after.stem) - target_timestamp_ns)

    if before_difference <= after_difference:
        return before

    return after


def ego_to_bev_pixels(x_ego: np.ndarray, y_ego: np.ndarray) -> np.ndarray:
    lateral_display = -y_ego

    u = (lateral_display - LATERAL_MIN) / RESOLUTION_M
    v = (X_MAX - x_ego) / RESOLUTION_M

    pixels = np.column_stack([u, v])
    return np.round(pixels).astype(np.int32)


def create_lidar_camera_bev_input(
    points: np.ndarray,
    scene: Path,
    lidar_timestamp_ns: int,
    city_SE3_ego_by_timestamp: dict,
    cameras_by_name: dict[str, PinholeCamera],
    camera_files_by_name: dict[str, list[Path]],
    camera_timestamps_by_name: dict[str, list[int]],
) -> np.ndarray:
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

    u_bev = np.round((lateral_display - LATERAL_MIN) / RESOLUTION_M).astype(np.int32)
    v_bev = np.round((X_MAX - x) / RESOLUTION_M).astype(np.int32)

    valid_bev_mask = (
        (u_bev >= 0) & (u_bev < WIDTH_PX) &
        (v_bev >= 0) & (v_bev < HEIGHT_PX)
    )

    x = x[valid_bev_mask]
    y = y[valid_bev_mask]
    z = z[valid_bev_mask]
    u_bev = u_bev[valid_bev_mask]
    v_bev = v_bev[valid_bev_mask]

    points_crop = np.column_stack([x, y, z]).astype(np.float64)

    density = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    height_sum = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    height_count = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    max_height = np.full((HEIGHT_PX, WIDTH_PX), fill_value=Z_MIN, dtype=np.float32)

    np.add.at(density, (v_bev, u_bev), 1.0)
    np.add.at(height_sum, (v_bev, u_bev), z)
    np.add.at(height_count, (v_bev, u_bev), 1.0)
    np.maximum.at(max_height, (v_bev, u_bev), z)

    mean_height = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    nonzero_height_mask = height_count > 0
    mean_height[nonzero_height_mask] = (
        height_sum[nonzero_height_mask] / height_count[nonzero_height_mask]
    )

    density_norm = np.log1p(density)

    if density_norm.max() > 0:
        density_norm = density_norm / density_norm.max()

    max_height_norm = (max_height - Z_MIN) / (Z_MAX - Z_MIN)
    max_height_norm = np.clip(max_height_norm, 0.0, 1.0)

    mean_height_norm = (mean_height - Z_MIN) / (Z_MAX - Z_MIN)
    mean_height_norm = np.clip(mean_height_norm, 0.0, 1.0)
    mean_height_norm[~nonzero_height_mask] = 0.0

    rgb_sum = np.zeros((HEIGHT_PX, WIDTH_PX, 3), dtype=np.float32)
    rgb_count = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)

    city_SE3_ego_lidar_t = city_SE3_ego_by_timestamp[lidar_timestamp_ns]

    for camera_name in RING_CAMERAS:
        camera_files = camera_files_by_name[camera_name]
        camera_timestamps = camera_timestamps_by_name[camera_name]

        camera_file = get_closest_file(
            files=camera_files,
            timestamps=camera_timestamps,
            target_timestamp_ns=lidar_timestamp_ns,
        )

        camera_timestamp_ns = int(camera_file.stem)

        if camera_timestamp_ns not in city_SE3_ego_by_timestamp:
            continue

        image = read_img(camera_file, channel_order="RGB")
        image_height, image_width = image.shape[:2]

        camera = cameras_by_name[camera_name]
        city_SE3_ego_camera_t = city_SE3_ego_by_timestamp[camera_timestamp_ns]

        uv, _, is_valid = camera.project_ego_to_img_motion_compensated(
            points_lidar_time=points_crop,
            city_SE3_ego_cam_t=city_SE3_ego_camera_t,
            city_SE3_ego_lidar_t=city_SE3_ego_lidar_t,
        )

        if not np.any(is_valid):
            continue

        uv_valid = uv[is_valid]
        u_bev_valid = u_bev[is_valid]
        v_bev_valid = v_bev[is_valid]

        u_img = np.round(uv_valid[:, 0]).astype(np.int32)
        v_img = np.round(uv_valid[:, 1]).astype(np.int32)

        inside_image_mask = (
            (u_img >= 0) & (u_img < image_width) &
            (v_img >= 0) & (v_img < image_height)
        )

        u_img = u_img[inside_image_mask]
        v_img = v_img[inside_image_mask]
        u_bev_valid = u_bev_valid[inside_image_mask]
        v_bev_valid = v_bev_valid[inside_image_mask]

        sampled_rgb = image[v_img, u_img, :].astype(np.float32) / 255.0

        np.add.at(rgb_sum[:, :, 0], (v_bev_valid, u_bev_valid), sampled_rgb[:, 0])
        np.add.at(rgb_sum[:, :, 1], (v_bev_valid, u_bev_valid), sampled_rgb[:, 1])
        np.add.at(rgb_sum[:, :, 2], (v_bev_valid, u_bev_valid), sampled_rgb[:, 2])
        np.add.at(rgb_count, (v_bev_valid, u_bev_valid), 1.0)

    rgb_mean = np.zeros_like(rgb_sum, dtype=np.float32)
    color_mask = rgb_count > 0

    rgb_mean[color_mask, 0] = rgb_sum[color_mask, 0] / rgb_count[color_mask]
    rgb_mean[color_mask, 1] = rgb_sum[color_mask, 1] / rgb_count[color_mask]
    rgb_mean[color_mask, 2] = rgb_sum[color_mask, 2] / rgb_count[color_mask]

    bev_input = np.stack(
        [
            density_norm,
            max_height_norm,
            mean_height_norm,
            rgb_mean[:, :, 0],
            rgb_mean[:, :, 1],
            rgb_mean[:, :, 2],
        ],
        axis=0,
    ).astype(np.float32)

    return bev_input


def create_bev_labels(
    static_map: ArgoverseStaticMap,
    ego_SE3_city,
) -> tuple[np.ndarray, np.ndarray]:
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
    print(f"BEV input shape: 6 by {HEIGHT_PX} by {WIDTH_PX}")

    for scene_index, scene in enumerate(scenes):
        print(f"\nProcessing scene {scene_index + 1}/{len(scenes)}: {scene.name}")

        lidar_dir = scene / "sensors" / "lidar"
        map_dir = scene / "map"

        lidar_files = sorted(lidar_dir.glob("*.feather"))
        lidar_files = lidar_files[::SAMPLE_STRIDE][:MAX_SAMPLES_PER_SCENE]

        city_SE3_ego_by_timestamp = read_city_SE3_ego(scene)

        map_json_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        static_map = ArgoverseStaticMap.from_json(map_json_path)

        cameras_by_name = {}
        camera_files_by_name = {}
        camera_timestamps_by_name = {}

        for camera_name in RING_CAMERAS:
            camera_dir = scene / "sensors" / "cameras" / camera_name
            camera_files = sorted(camera_dir.glob("*.jpg"))
            camera_timestamps = [int(path.stem) for path in camera_files]

            cameras_by_name[camera_name] = PinholeCamera.from_feather(
                log_dir=scene,
                cam_name=camera_name,
            )
            camera_files_by_name[camera_name] = camera_files
            camera_timestamps_by_name[camera_name] = camera_timestamps

        for lidar_file in tqdm(lidar_files, desc="Creating samples"):
            lidar_timestamp_ns = int(lidar_file.stem)

            if lidar_timestamp_ns not in city_SE3_ego_by_timestamp:
                continue

            points = read_lidar_sweep(lidar_file, attrib_spec="xyz")

            city_SE3_ego = city_SE3_ego_by_timestamp[lidar_timestamp_ns]
            ego_SE3_city = city_SE3_ego.inverse()

            bev_input = create_lidar_camera_bev_input(
                points=points,
                scene=scene,
                lidar_timestamp_ns=lidar_timestamp_ns,
                city_SE3_ego_by_timestamp=city_SE3_ego_by_timestamp,
                cameras_by_name=cameras_by_name,
                camera_files_by_name=camera_files_by_name,
                camera_timestamps_by_name=camera_timestamps_by_name,
            )

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
                    "input_channels": [
                        "lidar_density",
                        "lidar_max_height",
                        "lidar_mean_height",
                        "camera_red",
                        "camera_green",
                        "camera_blue",
                    ],
                    "camera_names": RING_CAMERAS,
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