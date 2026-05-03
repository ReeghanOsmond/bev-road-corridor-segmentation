from pathlib import Path
import json
import math
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm

from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import read_city_SE3_ego


DATA_ROOT = Path("data/av2/sensor/val")

SOURCE_DATASET_ROOT = Path("outputs/dataset_camera_color")
SOURCE_METADATA_PATH = SOURCE_DATASET_ROOT / "metadata.json"

OUTPUT_ROOT = Path("outputs/dataset_main_road_corridor")
LABEL_DIR = OUTPUT_ROOT / "labels_main_road"
BOUNDARY_LABEL_DIR = OUTPUT_ROOT / "labels_outer_boundaries"
FIGURE_DIR = Path("outputs/figures")

CLEAN_OUTPUT = True

X_MIN = -10.0
X_MAX = 60.0
LATERAL_MIN = -25.0
LATERAL_MAX = 25.0
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

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDARY_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

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
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned = keep_component_containing_ego_or_largest(cleaned)
    cleaned = fill_holes(cleaned)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    return cleaned


def create_outer_boundary_mask(corridor_mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    boundary = cv2.morphologyEx(corridor_mask, cv2.MORPH_GRADIENT, kernel)
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
        4,
        figsize=(16, 4 * number_of_examples),
    )

    if number_of_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, item in enumerate(preview_items[:number_of_examples]):
        bev_input = np.load(item["source_input_path"]).astype(np.float32)

        corridor_mask = cv2.imread(
            item["main_road_label_path"],
            cv2.IMREAD_GRAYSCALE,
        )

        boundary_mask = cv2.imread(
            item["outer_boundary_label_path"],
            cv2.IMREAD_GRAYSCALE,
        )

        density = bev_input[0]
        camera_rgb = np.moveaxis(bev_input[3:6], 0, 2)

        combined = np.full(
            (HEIGHT_PX, WIDTH_PX, 3),
            BACKGROUND_COLOR,
            dtype=np.uint8,
        )

        combined[corridor_mask > 0] = CORRIDOR_COLOR
        combined[boundary_mask > 0] = BOUNDARY_COLOR

        axes[row, 0].imshow(density, cmap="viridis")
        axes[row, 0].set_title(f"{item['sample_name']}\nLiDAR density")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(camera_rgb)
        axes[row, 1].set_title("Camera color in BEV")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(corridor_mask, cmap="gray")
        axes[row, 2].set_title("Main road corridor mask")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(combined)
        axes[row, 3].set_title("Corridor and outer boundary")
        axes[row, 3].axis("off")

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
    plt.suptitle("Main road corridor labels")
    plt.tight_layout(rect=[0.0, 0.05, 1.0, 0.97])

    figure_path = FIGURE_DIR / "main_road_corridor_label_examples.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved preview figure to: {figure_path}")


def main() -> None:
    reset_output_dirs()

    with SOURCE_METADATA_PATH.open("r") as f:
        source_metadata = json.load(f)

    scene_cache = {}
    output_metadata = []

    print(f"Source samples: {len(source_metadata)}")
    print(f"Output label size: {HEIGHT_PX} by {WIDTH_PX}")
    print(f"Lane alignment threshold: {HEADING_ALIGNMENT_DEGREES} degrees")

    for item in tqdm(source_metadata, desc="Creating main road labels"):
        scene_name = item["scene"]
        lidar_timestamp_ns = int(item["lidar_timestamp_ns"])

        if scene_name not in scene_cache:
            scene = DATA_ROOT / scene_name
            map_dir = scene / "map"
            map_json_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]

            scene_cache[scene_name] = {
                "scene_path": scene,
                "city_SE3_ego_by_timestamp": read_city_SE3_ego(scene),
                "static_map": ArgoverseStaticMap.from_json(map_json_path),
            }

        resources = scene_cache[scene_name]

        city_SE3_ego_by_timestamp = resources["city_SE3_ego_by_timestamp"]

        if lidar_timestamp_ns not in city_SE3_ego_by_timestamp:
            continue

        city_SE3_ego = city_SE3_ego_by_timestamp[lidar_timestamp_ns]
        ego_SE3_city = city_SE3_ego.inverse()

        corridor_mask, outer_boundary_mask = create_main_road_corridor_label(
            static_map=resources["static_map"],
            ego_SE3_city=ego_SE3_city,
        )

        if np.count_nonzero(corridor_mask) < MIN_MASK_PIXELS:
            continue

        sample_name = item["sample_name"]

        main_road_label_path = LABEL_DIR / f"{sample_name}.png"
        outer_boundary_label_path = BOUNDARY_LABEL_DIR / f"{sample_name}.png"

        cv2.imwrite(str(main_road_label_path), corridor_mask)
        cv2.imwrite(str(outer_boundary_label_path), outer_boundary_mask)

        output_metadata.append(
            {
                "sample_name": sample_name,
                "scene": scene_name,
                "lidar_timestamp_ns": lidar_timestamp_ns,
                "source_input_path": item["input_path"],
                "main_road_label_path": str(main_road_label_path),
                "outer_boundary_label_path": str(outer_boundary_label_path),
                "label_type": "ego_aligned_main_road_corridor",
                "input_channels": item["input_channels"],
                "x_min_m": X_MIN,
                "x_max_m": X_MAX,
                "lateral_min_m": LATERAL_MIN,
                "lateral_max_m": LATERAL_MAX,
                "resolution_m": RESOLUTION_M,
                "heading_alignment_degrees": HEADING_ALIGNMENT_DEGREES,
            }
        )

    metadata_path = OUTPUT_ROOT / "metadata.json"

    with metadata_path.open("w") as f:
        json.dump(output_metadata, f, indent=2)

    save_preview_figure(output_metadata)

    print("\nDone.")
    print(f"Created main road corridor samples: {len(output_metadata)}")
    print(f"Main road labels saved to: {LABEL_DIR}")
    print(f"Outer boundary labels saved to: {BOUNDARY_LABEL_DIR}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()