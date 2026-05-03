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


DATASET_ROOT = Path("outputs/dataset")
INPUT_DIR = DATASET_ROOT / "inputs"
LABEL_DIR = DATASET_ROOT / "labels_drivable"
METADATA_PATH = DATASET_ROOT / "metadata.json"

OUTPUT_DIR = Path("outputs/model_drivable")
FIGURE_DIR = Path("outputs/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class BevDrivableDataset(Dataset):
    def __init__(self, sample_names: list[str], augment: bool = False):
        self.sample_names = sample_names
        self.augment = augment

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        sample_name = self.sample_names[index]

        input_path = INPUT_DIR / f"{sample_name}.npy"
        label_path = LABEL_DIR / f"{sample_name}.png"

        bev_input = np.load(input_path).astype(np.float32)

        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        label = (label > 0).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                bev_input = np.flip(bev_input, axis=2).copy()
                label = np.flip(label, axis=1).copy()

            if random.random() < 0.3:
                noise = np.random.normal(loc=0.0, scale=0.02, size=bev_input.shape).astype(np.float32)
                bev_input = np.clip(bev_input + noise, 0.0, 1.0)

        bev_input = torch.from_numpy(bev_input)
        label = torch.from_numpy(label).unsqueeze(0)

        return bev_input, label, sample_name


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
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(3, 32)
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


def compute_metrics(logits, labels, threshold: float = 0.5):
    probabilities = torch.sigmoid(logits)
    predictions = probabilities > threshold
    labels_bool = labels > 0.5

    intersection = (predictions & labels_bool).sum().item()
    union = (predictions | labels_bool).sum().item()

    true_positive = intersection
    false_positive = (predictions & ~labels_bool).sum().item()
    false_negative = (~predictions & labels_bool).sum().item()

    iou = intersection / union if union > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0

    return iou, precision, recall


def load_scene_split():
    with METADATA_PATH.open("r") as f:
        metadata = json.load(f)

    scenes = sorted(set(item["scene"] for item in metadata))
    random.shuffle(scenes)

    split_index = max(1, int(0.8 * len(scenes)))
    train_scenes = set(scenes[:split_index])
    val_scenes = set(scenes[split_index:])

    train_samples = []
    val_samples = []

    for item in metadata:
        sample_name = item["sample_name"]

        if item["scene"] in train_scenes:
            train_samples.append(sample_name)
        else:
            val_samples.append(sample_name)

    return train_samples, val_samples, train_scenes, val_scenes


def save_prediction_examples(model, val_dataset, epoch_label: str):
    model.eval()

    number_of_examples = min(4, len(val_dataset))
    selected_indices = list(range(number_of_examples))

    fig, axes = plt.subplots(number_of_examples, 4, figsize=(16, 4 * number_of_examples))

    if number_of_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for row, index in enumerate(selected_indices):
            bev_input, label, sample_name = val_dataset[index]

            logits = model(bev_input.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits).squeeze().cpu().numpy()
            prediction = probability > 0.5

            input_density = bev_input[0].numpy()
            label_np = label.squeeze().numpy()

            axes[row, 0].imshow(input_density, cmap="viridis")
            axes[row, 0].set_title(f"{sample_name}\nInput density")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(label_np, cmap="gray")
            axes[row, 1].set_title("Ground truth")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(probability, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[row, 2].set_title("Prediction probability")
            axes[row, 2].axis("off")

            axes[row, 3].imshow(prediction, cmap="gray")
            axes[row, 3].set_title("Prediction mask")
            axes[row, 3].axis("off")

    plt.suptitle(f"Drivable area prediction examples, {epoch_label}")
    plt.tight_layout()

    output_path = FIGURE_DIR / f"drivable_area_prediction_examples_{epoch_label}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved prediction examples to: {output_path}")


def main():
    train_samples, val_samples, train_scenes, val_scenes = load_scene_split()

    print(f"Training scenes: {len(train_scenes)}")
    print(f"Validation scenes: {len(val_scenes)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    train_dataset = BevDrivableDataset(train_samples, augment=True)
    val_dataset = BevDrivableDataset(val_samples, augment=False)

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

    model = SmallUNet().to(device)

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

    best_val_iou = -1.0
    best_model_path = OUTPUT_DIR / "best_drivable_area_model.pt"
    final_model_path = OUTPUT_DIR / "final_drivable_area_model.pt"

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
        val_ious = []
        val_precisions = []
        val_recalls = []

        with torch.no_grad():
            for bev_inputs, labels, _ in val_loader:
                bev_inputs = bev_inputs.to(device)
                labels = labels.to(device)

                logits = model(bev_inputs)
                loss = combined_loss(logits, labels, bce_loss_function)

                val_loss_total += loss.item() * bev_inputs.size(0)

                iou, precision, recall = compute_metrics(logits, labels)
                val_ious.append(iou)
                val_precisions.append(precision)
                val_recalls.append(recall)

        val_loss = val_loss_total / len(val_dataset)
        mean_val_iou = float(np.mean(val_ious))
        mean_val_precision = float(np.mean(val_precisions))
        mean_val_recall = float(np.mean(val_recalls))
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": mean_val_iou,
            "val_precision": mean_val_precision,
            "val_recall": mean_val_recall,
            "learning_rate": current_lr,
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"IoU: {mean_val_iou:.4f} | "
            f"precision: {mean_val_precision:.4f} | "
            f"recall: {mean_val_recall:.4f} | "
            f"lr: {current_lr:.6f}"
        )

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_iou": best_val_iou,
                    "train_scenes": sorted(train_scenes),
                    "val_scenes": sorted(val_scenes),
                },
                best_model_path,
            )

        if (epoch + 1) in [1, 25, 50, 100, NUM_EPOCHS]:
            save_prediction_examples(model, val_dataset, epoch_label=f"epoch_{epoch + 1:03d}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": NUM_EPOCHS,
            "best_val_iou": best_val_iou,
            "train_scenes": sorted(train_scenes),
            "val_scenes": sorted(val_scenes),
        },
        final_model_path,
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
                "val_iou",
                "val_precision",
                "val_recall",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    val_ious = [item["val_iou"] for item in history]
    val_precisions = [item["val_precision"] for item in history]
    val_recalls = [item["val_recall"] for item in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Drivable area training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    loss_curve_path = FIGURE_DIR / "drivable_area_training_loss.png"
    plt.savefig(loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, val_ious, label="Validation IoU")
    plt.plot(epochs, val_precisions, label="Validation precision")
    plt.plot(epochs, val_recalls, label="Validation recall")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Drivable area validation metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)

    metrics_curve_path = FIGURE_DIR / "drivable_area_validation_metrics.png"
    plt.savefig(metrics_curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    save_prediction_examples(model, val_dataset, epoch_label="final")

    print("\nTraining complete.")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {metrics_json_path}")
    print(f"Training CSV saved to: {metrics_csv_path}")
    print(f"Loss curve saved to: {loss_curve_path}")
    print(f"Metrics curve saved to: {metrics_curve_path}")


if __name__ == "__main__":
    main()