from pathlib import Path
import bisect
import json
import math
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego, read_img, read_lidar_sweep


DATA_ROOT = Path("data/av2/sensor/val")
OUTPUT_ROOT = Path("outputs/dataset_main_road_corridor_position_features")

INPUT_DIR = OUTPUT_ROOT / "inputs"
LABEL_DIR = OUTPUT_ROOT / "labels_main_road"
BOUNDARY_LABEL_DIR = OUTPUT_ROOT / "labels_outer_boundaries"
FIGURE_DIR = Path("outputs/figures")

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

HEADING_ALIGNMENT_DEGREES = 35.0
MIN_ALIGNMENT = math.cos(math.radians(HEADING_ALIGNMENT_DEGREES))

MIN_MASK_PIXELS = 250

CORRIDOR_COLOR = np.array([86, 180, 233], dtype=np.uint8)
BOUNDARY_COLOR = np.array([213, 94, 0], dtype=np.uint8)
BACKGROUND_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def reset_output_dirs() -> None:
    if CLEAN_OUTPUT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDARY_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def get_closest_file(
    files: list[Path],
    timestamps: list[int],
    target_timestamp_ns: int,
) -> Path:
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


def crop_points_to_bev(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    valid_pixel_mask = (
        (u_bev >= 0) & (u_bev < WIDTH_PX) &
        (v_bev >= 0) & (v_bev < HEIGHT_PX)
    )

    points_crop = np.column_stack(
        [
            x[valid_pixel_mask],
            y[valid_pixel_mask],
            z[valid_pixel_mask],
        ]
    ).astype(np.float64)

    u_bev = u_bev[valid_pixel_mask]
    v_bev = v_bev[valid_pixel_mask]

    return points_crop, u_bev, v_bev


def create_lidar_bev_features(
    points_crop: np.ndarray,
    u_bev: np.ndarray,
    v_bev: np.ndarray,
) -> np.ndarray:
    z = points_crop[:, 2]

    density = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    height_sum = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    height_count = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    max_height = np.full((HEIGHT_PX, WIDTH_PX), fill_value=Z_MIN, dtype=np.float32)

    np.add.at(density, (v_bev, u_bev), 1.0)
    np.add.at(height_sum, (v_bev, u_bev), z)
    np.add.at(height_count, (v_bev, u_bev), 1.0)
    np.maximum.at(max_height, (v_bev, u_bev), z)

    has_points = height_count > 0

    mean_height = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    mean_height[has_points] = height_sum[has_points] / height_count[has_points]

    density_norm = np.log1p(density)

    if density_norm.max() > 0:
        density_norm = density_norm / density_norm.max()

    max_height_norm = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    max_height_norm[has_points] = (
        (max_height[has_points] - Z_MIN) /
        (Z_MAX - Z_MIN)
    )
    max_height_norm = np.clip(max_height_norm, 0.0, 1.0)

    mean_height_norm = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    mean_height_norm[has_points] = (
        (mean_height[has_points] - Z_MIN) /
        (Z_MAX - Z_MIN)
    )
    mean_height_norm = np.clip(mean_height_norm, 0.0, 1.0)

    lidar_features = np.stack(
        [
            density_norm,
            max_height_norm,
            mean_height_norm,
        ],
        axis=0,
    ).astype(np.float32)

    return lidar_features


def create_camera_color_features(
    points_crop: np.ndarray,
    u_bev: np.ndarray,
    v_bev: np.ndarray,
    lidar_timestamp_ns: int,
    city_SE3_ego_by_timestamp: dict,
    cameras_by_name: dict[str, PinholeCamera],
    camera_files_by_name: dict[str, list[Path]],
    camera_timestamps_by_name: dict[str, list[int]],
) -> np.ndarray:
    red_sum = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    green_sum = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    blue_sum = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.float32)
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

        np.add.at(red_sum, (v_bev_valid, u_bev_valid), sampled_rgb[:, 0])
        np.add.at(green_sum, (v_bev_valid, u_bev_valid), sampled_rgb[:, 1])
        np.add.at(blue_sum, (v_bev_valid, u_bev_valid), sampled_rgb[:, 2])
        np.add.at(rgb_count, (v_bev_valid, u_bev_valid), 1.0)

    camera_rgb = np.zeros((3, HEIGHT_PX, WIDTH_PX), dtype=np.float32)
    has_color = rgb_count > 0

    camera_rgb[0, has_color] = red_sum[has_color] / rgb_count[has_color]
    camera_rgb[1, has_color] = green_sum[has_color] / rgb_count[has_color]
    camera_rgb[2, has_color] = blue_sum[has_color] / rgb_count[has_color]

    return camera_rgb


def create_position_features() -> np.ndarray:
    rows, cols = np.indices((HEIGHT_PX, WIDTH_PX))

    x_forward = X_MAX - (rows.astype(np.float32) + 0.5) * RESOLUTION_M
    lateral_display = LATERAL_MIN + (cols.astype(np.float32) + 0.5) * RESOLUTION_M

    y_ego = -lateral_display

    distance = np.sqrt(x_forward ** 2 + y_ego ** 2)
    max_distance = np.sqrt(
        max(abs(X_MIN), abs(X_MAX)) ** 2 +
        max(abs(LATERAL_MIN), abs(LATERAL_MAX)) ** 2
    )

    distance_norm = np.clip(distance / max_distance, 0.0, 1.0)

    sin_angle = np.zeros_like(distance, dtype=np.float32)
    cos_angle = np.zeros_like(distance, dtype=np.float32)

    nonzero_distance = distance > 1e-6

    sin_angle[nonzero_distance] = y_ego[nonzero_distance] / distance[nonzero_distance]
    cos_angle[nonzero_distance] = x_forward[nonzero_distance] / distance[nonzero_distance]

    forward_norm = (x_forward - X_MIN) / (X_MAX - X_MIN)
    forward_norm = np.clip(forward_norm, 0.0, 1.0)

    lateral_norm = (lateral_display - LATERAL_MIN) / (LATERAL_MAX - LATERAL_MIN)
    lateral_norm = np.clip(lateral_norm, 0.0, 1.0)

    position_features = np.stack(
        [
            distance_norm,
            sin_angle,
            cos_angle,
            forward_norm,
            lateral_norm,
        ],
        axis=0,
    ).astype(np.float32)

    return position_features


def ego_to_bev_pixels(x_ego: np.ndarray, y_ego: np.ndarray) -> np.ndarray:
    lateral_display = -y_ego

    u = (lateral_display - LATERAL_MIN) / RESOLUTION_M
    v = (X_MAX - x_ego) / RESOLUTION_M

    pixels = np.column_stack([u, v])
    return np.round(pixels).astype(np.int32)


def lane_alignment_with_ego_forward(left_xy: np.ndarray, right_xy: np.ndarray) -> float:
    points = np.vstack([left_xy, right_xy])

    if points.shape[0] < 2:
        return 0.0

    centered = points - points.mean(axis=0, keepdims=True)

    if np.allclose(centered, 0.0):
        return 0.0

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]

    norm = np.linalg.norm(direction)

    if norm == 0.0:
        return 0.0

    direction = direction / norm

    return abs(direction[0])


def make_lane_polygon(left_xy: np.ndarray, right_xy: np.ndarray) -> np.ndarray:
    if np.linalg.norm(left_xy[0] - right_xy[0]) > np.linalg.norm(left_xy[0] - right_xy[-1]):
        right_xy = right_xy[::-1]

    polygon_xy = np.vstack([left_xy, right_xy[::-1]])
    return polygon_xy


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    inverse = (binary == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(inverse, connectivity=8)

    border_labels = set(labels[0, :])
    border_labels.update(labels[-1, :])
    border_labels.update(labels[:, 0])
    border_labels.update(labels[:, -1])

    filled = binary.copy()

    for label_id in range(1, num_labels):
        if label_id not in border_labels:
            filled[labels == label_id] = 1

    return (filled * 255).astype(np.uint8)


def keep_component_containing_ego_or_largest(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask

    ego_pixel = ego_to_bev_pixels(
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )[0]

    ego_u = int(ego_pixel[0])
    ego_v = int(ego_pixel[1])

    keep_label = 0

    if 0 <= ego_u < WIDTH_PX and 0 <= ego_v < HEIGHT_PX:
        ego_label = labels[ego_v, ego_u]

        if ego_label > 0:
            keep_label = ego_label

    if keep_label == 0:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        keep_label = int(np.argmax(component_areas)) + 1

    cleaned = np.zeros_like(mask)
    cleaned[labels == keep_label] = 255

    return cleaned


def clean_corridor_mask(mask: np.ndarray) -> np.ndarray:
    cleaned = keep_component_containing_ego_or_largest(mask)
    cleaned = fill_holes(cleaned)

    return cleaned


def create_outer_boundary_mask(corridor_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        corridor_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boundary = np.zeros_like(corridor_mask)

    cv2.drawContours(
        boundary,
        contours,
        contourIdx=-1,
        color=255,
        thickness=2,
    )

    return boundary


def create_main_road_corridor_label(
    static_map: ArgoverseStaticMap,
    ego_SE3_city,
) -> tuple[np.ndarray, np.ndarray]:
    raw_corridor_mask = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.uint8)

    for lane_segment in static_map.vector_lane_segments.values():
        left_xyz_city = lane_segment.left_lane_boundary.xyz
        right_xyz_city = lane_segment.right_lane_boundary.xyz

        left_xyz_ego = ego_SE3_city.transform_point_cloud(left_xyz_city)
        right_xyz_ego = ego_SE3_city.transform_point_cloud(right_xyz_city)

        left_xy = left_xyz_ego[:, :2]
        right_xy = right_xyz_ego[:, :2]

        all_x = np.concatenate([left_xy[:, 0], right_xy[:, 0]])
        all_y = np.concatenate([left_xy[:, 1], right_xy[:, 1]])

        if (
            np.max(all_x) < X_MIN or np.min(all_x) > X_MAX or
            np.max(-all_y) < LATERAL_MIN or np.min(-all_y) > LATERAL_MAX
        ):
            continue

        alignment = lane_alignment_with_ego_forward(left_xy, right_xy)

        if alignment < MIN_ALIGNMENT:
            continue

        polygon_xy = make_lane_polygon(left_xy, right_xy)

        polygon_pixels = ego_to_bev_pixels(
            polygon_xy[:, 0],
            polygon_xy[:, 1],
        )

        cv2.fillPoly(
            raw_corridor_mask,
            [polygon_pixels],
            color=255,
        )

    corridor_mask = clean_corridor_mask(raw_corridor_mask)
    outer_boundary_mask = create_outer_boundary_mask(corridor_mask)

    return corridor_mask, outer_boundary_mask


def save_preview_figure(preview_items: list[dict]) -> None:
    number_of_examples = min(4, len(preview_items))

    if number_of_examples == 0:
        return

    fig, axes = plt.subplots(
        number_of_examples,
        8,
        figsize=(30, 4 * number_of_examples),
    )

    if number_of_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, item in enumerate(preview_items[:number_of_examples]):
        bev_input = np.load(item["input_path"]).astype(np.float32)

        corridor_mask = cv2.imread(
            item["main_road_label_path"],
            cv2.IMREAD_GRAYSCALE,
        )

        boundary_mask = cv2.imread(
            item["outer_boundary_label_path"],
            cv2.IMREAD_GRAYSCALE,
        )

        camera_rgb = np.moveaxis(bev_input[3:6], 0, 2)

        combined = np.full(
            (HEIGHT_PX, WIDTH_PX, 3),
            BACKGROUND_COLOR,
            dtype=np.uint8,
        )

        combined[corridor_mask > 0] = CORRIDOR_COLOR
        combined[boundary_mask > 0] = BOUNDARY_COLOR

        axes[row, 0].imshow(bev_input[0], cmap="viridis")
        axes[row, 0].set_title(f"{item['sample_name']}\nLiDAR density")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(camera_rgb)
        axes[row, 1].set_title("Camera color")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(bev_input[6], cmap="viridis")
        axes[row, 2].set_title("Distance from ego")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(bev_input[7], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[row, 3].set_title("sin angle")
        axes[row, 3].axis("off")

        axes[row, 4].imshow(bev_input[8], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[row, 4].set_title("cos angle")
        axes[row, 4].axis("off")

        axes[row, 5].imshow(bev_input[9], cmap="viridis")
        axes[row, 5].set_title("Forward position")
        axes[row, 5].axis("off")

        axes[row, 6].imshow(bev_input[10], cmap="viridis")
        axes[row, 6].set_title("Lateral position")
        axes[row, 6].axis("off")

        axes[row, 7].imshow(combined)
        axes[row, 7].set_title("Target label")
        axes[row, 7].axis("off")

    legend_handles = [
        Patch(
            facecolor=CORRIDOR_COLOR / 255.0,
            edgecolor="none",
            label="Main road corridor",
        ),
        Patch(
            facecolor=BOUNDARY_COLOR / 255.0,
            edgecolor="none",
            label="Outer boundary",
        ),
    ]

    fig.legend(handles=legend_handles, loc="lower center", ncol=2)
    plt.suptitle("Position feature BEV inputs and main road corridor labels")
    plt.tight_layout(rect=[0.0, 0.05, 1.0, 0.97])

    figure_path = FIGURE_DIR / "position_main_road_corridor_dataset_examples.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved preview figure to: {figure_path}")


def main() -> None:
    reset_output_dirs()

    scenes = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])[:MAX_SCENES]

    position_features = create_position_features()

    output_metadata = []
    global_sample_index = 0

    print(f"Scenes selected: {len(scenes)}")
    print(f"Maximum samples per scene: {MAX_SAMPLES_PER_SCENE}")
    print(f"Sample stride: {SAMPLE_STRIDE}")
    print(f"Output input size: 11 by {HEIGHT_PX} by {WIDTH_PX}")

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

            points_crop, u_bev, v_bev = crop_points_to_bev(points)

            if len(points_crop) == 0:
                continue

            lidar_features = create_lidar_bev_features(
                points_crop=points_crop,
                u_bev=u_bev,
                v_bev=v_bev,
            )

            camera_rgb_features = create_camera_color_features(
                points_crop=points_crop,
                u_bev=u_bev,
                v_bev=v_bev,
                lidar_timestamp_ns=lidar_timestamp_ns,
                city_SE3_ego_by_timestamp=city_SE3_ego_by_timestamp,
                cameras_by_name=cameras_by_name,
                camera_files_by_name=camera_files_by_name,
                camera_timestamps_by_name=camera_timestamps_by_name,
            )

            bev_input = np.concatenate(
                [
                    lidar_features,
                    camera_rgb_features,
                    position_features,
                ],
                axis=0,
            ).astype(np.float32)

            city_SE3_ego = city_SE3_ego_by_timestamp[lidar_timestamp_ns]
            ego_SE3_city = city_SE3_ego.inverse()

            corridor_mask, outer_boundary_mask = create_main_road_corridor_label(
                static_map=static_map,
                ego_SE3_city=ego_SE3_city,
            )

            if np.count_nonzero(corridor_mask) < MIN_MASK_PIXELS:
                continue

            sample_name = f"sample_{global_sample_index:06d}"

            input_path = INPUT_DIR / f"{sample_name}.npy"
            main_road_label_path = LABEL_DIR / f"{sample_name}.png"
            outer_boundary_label_path = BOUNDARY_LABEL_DIR / f"{sample_name}.png"

            np.save(input_path, bev_input)
            cv2.imwrite(str(main_road_label_path), corridor_mask)
            cv2.imwrite(str(outer_boundary_label_path), outer_boundary_mask)

            output_metadata.append(
                {
                    "sample_name": sample_name,
                    "scene_index": scene_index,
                    "scene": scene.name,
                    "lidar_timestamp_ns": lidar_timestamp_ns,
                    "lidar_file": str(lidar_file),
                    "input_path": str(input_path),
                    "main_road_label_path": str(main_road_label_path),
                    "outer_boundary_label_path": str(outer_boundary_label_path),
                    "label_type": "ego_aligned_main_road_corridor",
                    "input_channels": [
                        "lidar_point_density",
                        "lidar_max_height",
                        "lidar_mean_height",
                        "camera_red",
                        "camera_green",
                        "camera_blue",
                        "ego_distance",
                        "sin_angle",
                        "cos_angle",
                        "forward_position",
                        "lateral_position",
                    ],
                    "x_min_m": X_MIN,
                    "x_max_m": X_MAX,
                    "lateral_min_m": LATERAL_MIN,
                    "lateral_max_m": LATERAL_MAX,
                    "resolution_m": RESOLUTION_M,
                    "heading_alignment_degrees": HEADING_ALIGNMENT_DEGREES,
                }
            )

            global_sample_index += 1

    metadata_path = OUTPUT_ROOT / "metadata.json"

    with metadata_path.open("w") as f:
        json.dump(output_metadata, f, indent=2)

    save_preview_figure(output_metadata)

    print("\nDone.")
    print(f"Created samples: {len(output_metadata)}")
    print(f"Inputs saved to: {INPUT_DIR}")
    print(f"Main road labels saved to: {LABEL_DIR}")
    print(f"Outer boundary labels saved to: {BOUNDARY_LABEL_DIR}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()