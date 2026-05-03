from pathlib import Path
import csv
import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT = Path("outputs/dataset_main_road_corridor_position_features")
METADATA_PATH = DATASET_ROOT / "metadata.json"

OUTPUT_DIR = Path("outputs/model_main_road_corridor_position_features")
FIGURE_DIR = Path("outputs/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.2

INPUT_CHANNELS = 11

X_MAX = 60.0
RESOLUTION_M = 0.25

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MainRoadPositionDataset(Dataset):
    def __init__(self, items: list[dict], augment: bool = False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        bev_input = np.load(item["input_path"]).astype(np.float32)

        label = cv2.imread(
            item["main_road_label_path"],
            cv2.IMREAD_GRAYSCALE,
        )

        label = (label > 0).astype(np.float32)

        if self.augment:
            if random.random() < 0.3:
                noise = np.random.normal(
                    loc=0.0,
                    scale=0.02,
                    size=bev_input[:6].shape,
                ).astype(np.float32)

                bev_input[:6] = np.clip(bev_input[:6] + noise, 0.0, 1.0)

        bev_input = torch.from_numpy(bev_input)
        label = torch.from_numpy(label).unsqueeze(0)

        return bev_input, label, item["sample_name"]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = INPUT_CHANNELS):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = DoubleConv(128, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(64 + 32, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        middle = self.middle(self.pool3(enc3))

        x = self.up3(middle)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        return self.out(x)


def dice_loss(logits, labels, smooth: float = 1.0):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(probabilities.size(0), -1)
    labels = labels.view(labels.size(0), -1)

    intersection = (probabilities * labels).sum(dim=1)
    denominator = probabilities.sum(dim=1) + labels.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def combined_loss(logits, labels, bce_loss_function):
    bce = bce_loss_function(logits, labels)
    dice = dice_loss(logits, labels)

    return bce + dice


def binary_metrics_numpy(prediction: np.ndarray, label: np.ndarray) -> tuple[float, float, float]:
    prediction = prediction.astype(bool)
    label = label.astype(bool)

    intersection = np.logical_and(prediction, label).sum()
    union = np.logical_or(prediction, label).sum()

    true_positive = intersection
    false_positive = np.logical_and(prediction, ~label).sum()
    false_negative = np.logical_and(~prediction, label).sum()

    iou = intersection / union if union > 0 else 0.0

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    return float(iou), float(precision), float(recall)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)

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

    return filled.astype(bool)


def keep_component_containing_ego_or_largest(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask.astype(bool)

    height, width = binary.shape

    ego_u = width // 2
    ego_v = int(round((X_MAX - 0.0) / RESOLUTION_M))

    keep_label = 0

    if 0 <= ego_u < width and 0 <= ego_v < height:
        ego_label = labels[ego_v, ego_u]

        if ego_label > 0:
            keep_label = ego_label

    if keep_label == 0:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        keep_label = int(np.argmax(component_areas)) + 1

    cleaned = np.zeros_like(binary)
    cleaned[labels == keep_label] = 1

    return cleaned.astype(bool)


def postprocess_prediction_mask(prediction_mask: np.ndarray) -> np.ndarray:
    cleaned = keep_component_containing_ego_or_largest(prediction_mask)
    cleaned = fill_holes(cleaned)

    return cleaned.astype(bool)


def compute_batch_metrics(
    logits,
    labels,
    threshold: float = 0.5,
    postprocess: bool = False,
) -> tuple[float, float, float]:
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy() > 0.5

    ious = []
    precisions = []
    recalls = []

    for batch_index in range(probabilities.shape[0]):
        prediction_mask = probabilities[batch_index, 0] > threshold

        if postprocess:
            prediction_mask = postprocess_prediction_mask(prediction_mask)

        label_mask = labels_np[batch_index, 0]

        iou, precision, recall = binary_metrics_numpy(
            prediction=prediction_mask,
            label=label_mask,
        )

        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)

    return float(np.mean(ious)), float(np.mean(precisions)), float(np.mean(recalls))


def load_random_sample_split():
    with METADATA_PATH.open("r") as f:
        metadata = json.load(f)

    random.shuffle(metadata)

    split_index = int((1.0 - VAL_FRACTION) * len(metadata))

    train_items = metadata[:split_index]
    val_items = metadata[split_index:]

    train_scenes = sorted(set(item["scene"] for item in train_items))
    val_scenes = sorted(set(item["scene"] for item in val_items))

    split_info = {
        "split_type": "random_sample_split",
        "val_fraction": VAL_FRACTION,
        "random_seed": RANDOM_SEED,
        "total_samples": len(metadata),
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "train_scene_count": len(train_scenes),
        "val_scene_count": len(val_scenes),
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "train_sample_names": [item["sample_name"] for item in train_items],
        "val_sample_names": [item["sample_name"] for item in val_items],
    }

    split_path = OUTPUT_DIR / "random_sample_split.json"

    with split_path.open("w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split details to: {split_path}")

    return train_items, val_items, split_info


def save_prediction_examples(model, val_dataset, epoch_label: str):
    model.eval()

    number_of_examples = min(4, len(val_dataset))
    selected_indices = list(range(number_of_examples))

    fig, axes = plt.subplots(
        number_of_examples,
        8,
        figsize=(30, 4 * number_of_examples),
    )

    if number_of_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row, index in enumerate(selected_indices):
            bev_input, label, sample_name = val_dataset[index]

            logits = model(bev_input.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits).squeeze().cpu().numpy()

            raw_prediction = probability > 0.5
            postprocessed_prediction = postprocess_prediction_mask(raw_prediction)

            input_density = bev_input[0].numpy()
            camera_rgb = np.moveaxis(bev_input[3:6].numpy(), 0, 2)
            distance = bev_input[6].numpy()
            sin_angle = bev_input[7].numpy()
            cos_angle = bev_input[8].numpy()
            label_np = label.squeeze().numpy() > 0.5

            false_positive = postprocessed_prediction & ~label_np
            false_negative = ~postprocessed_prediction & label_np

            error_rgb = np.zeros((*label_np.shape, 3), dtype=np.float32)
            error_rgb[label_np] = [0.6, 0.6, 0.6]
            error_rgb[false_positive] = [0.84, 0.37, 0.0]
            error_rgb[false_negative] = [0.0, 0.45, 0.70]

            axes[row, 0].imshow(input_density, cmap="viridis")
            axes[row, 0].set_title(f"{sample_name}\nLiDAR density")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(camera_rgb)
            axes[row, 1].set_title("Camera color")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(distance, cmap="viridis")
            axes[row, 2].set_title("Distance from ego")
            axes[row, 2].axis("off")

            axes[row, 3].imshow(sin_angle, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[row, 3].set_title("sin angle")
            axes[row, 3].axis("off")

            axes[row, 4].imshow(cos_angle, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            axes[row, 4].set_title("cos angle")
            axes[row, 4].axis("off")

            axes[row, 5].imshow(label_np, cmap="gray")
            axes[row, 5].set_title("Ground truth")
            axes[row, 5].axis("off")

            axes[row, 6].imshow(probability, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[row, 6].set_title("Probability")
            axes[row, 6].axis("off")

            axes[row, 7].imshow(error_rgb)
            axes[row, 7].set_title("Postprocessed errors\norange FP, blue FN")
            axes[row, 7].axis("off")

    plt.suptitle(f"Position feature main road corridor prediction examples, {epoch_label}")
    plt.tight_layout()

    output_path = FIGURE_DIR / f"position_main_road_corridor_prediction_examples_{epoch_label}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved prediction examples to: {output_path}")


def save_checkpoint(
    model,
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    raw_val_iou: float,
    raw_val_precision: float,
    raw_val_recall: float,
    post_val_iou: float,
    post_val_precision: float,
    post_val_recall: float,
    split_info: dict,
    checkpoint_type: str,
):
    torch.save(
        {
            "checkpoint_type": checkpoint_type,
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "raw_val_iou": raw_val_iou,
            "raw_val_precision": raw_val_precision,
            "raw_val_recall": raw_val_recall,
            "postprocessed_val_iou": post_val_iou,
            "postprocessed_val_precision": post_val_precision,
            "postprocessed_val_recall": post_val_recall,
            "split_info": split_info,
        },
        path,
    )


def main():
    train_items, val_items, split_info = load_random_sample_split()

    print(f"Split type: random sample split")
    print(f"Training samples: {len(train_items)}")
    print(f"Validation samples: {len(val_items)}")
    print(f"Training scenes represented: {split_info['train_scene_count']}")
    print(f"Validation scenes represented: {split_info['val_scene_count']}")

    train_dataset = MainRoadPositionDataset(train_items, augment=True)
    val_dataset = MainRoadPositionDataset(val_items, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SmallUNet(in_channels=INPUT_CHANNELS).to(device)

    bce_loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    history = []

    best_raw_val_iou = -1.0
    best_val_loss = float("inf")
    best_postprocessed_val_iou = -1.0

    best_raw_iou_epoch = 0
    best_loss_epoch = 0
    best_postprocessed_iou_epoch = 0

    best_raw_iou_model_path = OUTPUT_DIR / "best_position_model_by_raw_iou.pt"
    best_loss_model_path = OUTPUT_DIR / "best_position_model_by_loss.pt"
    best_postprocessed_iou_model_path = OUTPUT_DIR / "best_position_model_by_postprocessed_iou.pt"
    final_model_path = OUTPUT_DIR / "final_position_model.pt"

    for epoch in range(NUM_EPOCHS):
        model.train()

        train_loss_total = 0.0

        for bev_inputs, labels, _ in train_loader:
            bev_inputs = bev_inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(bev_inputs)
            loss = combined_loss(logits, labels, bce_loss_function)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * bev_inputs.size(0)

        scheduler.step()

        train_loss = train_loss_total / len(train_dataset)

        model.eval()

        val_loss_total = 0.0

        raw_ious = []
        raw_precisions = []
        raw_recalls = []

        post_ious = []
        post_precisions = []
        post_recalls = []

        with torch.no_grad():
            for bev_inputs, labels, _ in val_loader:
                bev_inputs = bev_inputs.to(device)
                labels = labels.to(device)

                logits = model(bev_inputs)
                loss = combined_loss(logits, labels, bce_loss_function)

                val_loss_total += loss.item() * bev_inputs.size(0)

                raw_iou, raw_precision, raw_recall = compute_batch_metrics(
                    logits=logits,
                    labels=labels,
                    threshold=0.5,
                    postprocess=False,
                )

                post_iou, post_precision, post_recall = compute_batch_metrics(
                    logits=logits,
                    labels=labels,
                    threshold=0.5,
                    postprocess=True,
                )

                raw_ious.append(raw_iou)
                raw_precisions.append(raw_precision)
                raw_recalls.append(raw_recall)

                post_ious.append(post_iou)
                post_precisions.append(post_precision)
                post_recalls.append(post_recall)

        val_loss = val_loss_total / len(val_dataset)

        mean_raw_val_iou = float(np.mean(raw_ious))
        mean_raw_val_precision = float(np.mean(raw_precisions))
        mean_raw_val_recall = float(np.mean(raw_recalls))

        mean_post_val_iou = float(np.mean(post_ious))
        mean_post_val_precision = float(np.mean(post_precisions))
        mean_post_val_recall = float(np.mean(post_recalls))

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_number = epoch + 1

        epoch_record = {
            "epoch": epoch_number,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "raw_val_iou": mean_raw_val_iou,
            "raw_val_precision": mean_raw_val_precision,
            "raw_val_recall": mean_raw_val_recall,
            "postprocessed_val_iou": mean_post_val_iou,
            "postprocessed_val_precision": mean_post_val_precision,
            "postprocessed_val_recall": mean_post_val_recall,
            "learning_rate": current_lr,
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch_number:03d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"raw IoU: {mean_raw_val_iou:.4f} | "
            f"post IoU: {mean_post_val_iou:.4f} | "
            f"raw precision: {mean_raw_val_precision:.4f} | "
            f"post precision: {mean_post_val_precision:.4f} | "
            f"raw recall: {mean_raw_val_recall:.4f} | "
            f"post recall: {mean_post_val_recall:.4f} | "
            f"lr: {current_lr:.6f}"
        )

        if mean_raw_val_iou > best_raw_val_iou:
            best_raw_val_iou = mean_raw_val_iou
            best_raw_iou_epoch = epoch_number

            save_checkpoint(
                model=model,
                path=best_raw_iou_model_path,
                epoch=epoch_number,
                train_loss=train_loss,
                val_loss=val_loss,
                raw_val_iou=mean_raw_val_iou,
                raw_val_precision=mean_raw_val_precision,
                raw_val_recall=mean_raw_val_recall,
                post_val_iou=mean_post_val_iou,
                post_val_precision=mean_post_val_precision,
                post_val_recall=mean_post_val_recall,
                split_info=split_info,
                checkpoint_type="best_raw_iou",
            )

            save_prediction_examples(
                model,
                val_dataset,
                epoch_label="best_raw_iou",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch_number

            save_checkpoint(
                model=model,
                path=best_loss_model_path,
                epoch=epoch_number,
                train_loss=train_loss,
                val_loss=val_loss,
                raw_val_iou=mean_raw_val_iou,
                raw_val_precision=mean_raw_val_precision,
                raw_val_recall=mean_raw_val_recall,
                post_val_iou=mean_post_val_iou,
                post_val_precision=mean_post_val_precision,
                post_val_recall=mean_post_val_recall,
                split_info=split_info,
                checkpoint_type="best_loss",
            )

            save_prediction_examples(
                model,
                val_dataset,
                epoch_label="best_loss",
            )

        if mean_post_val_iou > best_postprocessed_val_iou:
            best_postprocessed_val_iou = mean_post_val_iou
            best_postprocessed_iou_epoch = epoch_number

            save_checkpoint(
                model=model,
                path=best_postprocessed_iou_model_path,
                epoch=epoch_number,
                train_loss=train_loss,
                val_loss=val_loss,
                raw_val_iou=mean_raw_val_iou,
                raw_val_precision=mean_raw_val_precision,
                raw_val_recall=mean_raw_val_recall,
                post_val_iou=mean_post_val_iou,
                post_val_precision=mean_post_val_precision,
                post_val_recall=mean_post_val_recall,
                split_info=split_info,
                checkpoint_type="best_postprocessed_iou",
            )

            save_prediction_examples(
                model,
                val_dataset,
                epoch_label="best_postprocessed_iou",
            )

        if epoch_number in [1, 25, 50, 100, NUM_EPOCHS]:
            save_prediction_examples(
                model,
                val_dataset,
                epoch_label=f"epoch_{epoch_number:03d}",
            )

    save_checkpoint(
        model=model,
        path=final_model_path,
        epoch=NUM_EPOCHS,
        train_loss=history[-1]["train_loss"],
        val_loss=history[-1]["val_loss"],
        raw_val_iou=history[-1]["raw_val_iou"],
        raw_val_precision=history[-1]["raw_val_precision"],
        raw_val_recall=history[-1]["raw_val_recall"],
        post_val_iou=history[-1]["postprocessed_val_iou"],
        post_val_precision=history[-1]["postprocessed_val_precision"],
        post_val_recall=history[-1]["postprocessed_val_recall"],
        split_info=split_info,
        checkpoint_type="final",
    )

    metrics_json_path = OUTPUT_DIR / "training_history.json"

    with metrics_json_path.open("w") as f:
        json.dump(history, f, indent=2)

    metrics_csv_path = OUTPUT_DIR / "training_history.csv"

    with metrics_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "raw_val_iou",
                "raw_val_precision",
                "raw_val_recall",
                "postprocessed_val_iou",
                "postprocessed_val_precision",
                "postprocessed_val_recall",
                "learning_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    raw_ious = [item["raw_val_iou"] for item in history]
    post_ious = [item["postprocessed_val_iou"] for item in history]
    raw_precisions = [item["raw_val_precision"] for item in history]
    post_precisions = [item["postprocessed_val_precision"] for item in history]
    raw_recalls = [item["raw_val_recall"] for item in history]
    post_recalls = [item["postprocessed_val_recall"] for item in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Position feature main road corridor training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    loss_curve_path = FIGURE_DIR / "position_main_road_corridor_training_loss.png"
    plt.savefig(loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, raw_ious, label="Raw validation IoU")
    plt.plot(epochs, post_ious, label="Postprocessed validation IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Position feature main road corridor IoU")
    plt.legend()
    plt.grid(True, alpha=0.3)

    iou_curve_path = FIGURE_DIR / "position_main_road_corridor_iou.png"
    plt.savefig(iou_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, raw_precisions, label="Raw precision")
    plt.plot(epochs, post_precisions, label="Postprocessed precision")
    plt.plot(epochs, raw_recalls, label="Raw recall")
    plt.plot(epochs, post_recalls, label="Postprocessed recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Position feature main road corridor precision and recall")
    plt.legend()
    plt.grid(True, alpha=0.3)

    metrics_curve_path = FIGURE_DIR / "position_main_road_corridor_precision_recall.png"
    plt.savefig(metrics_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    save_prediction_examples(model, val_dataset, epoch_label="final")

    print("\nTraining complete.")
    print(f"Best raw validation IoU: {best_raw_val_iou:.4f} at epoch {best_raw_iou_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_loss_epoch}")
    print(
        f"Best postprocessed validation IoU: "
        f"{best_postprocessed_val_iou:.4f} at epoch {best_postprocessed_iou_epoch}"
    )
    print(f"Best raw IoU model saved to: {best_raw_iou_model_path}")
    print(f"Best loss model saved to: {best_loss_model_path}")
    print(f"Best postprocessed IoU model saved to: {best_postprocessed_iou_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {metrics_json_path}")
    print(f"Training CSV saved to: {metrics_csv_path}")
    print(f"Loss curve saved to: {loss_curve_path}")
    print(f"IoU curve saved to: {iou_curve_path}")
    print(f"Precision recall curve saved to: {metrics_curve_path}")


if __name__ == "__main__":
    main()