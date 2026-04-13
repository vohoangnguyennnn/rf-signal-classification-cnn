import copy
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision import transforms


def lock_every_seed(seed_value=42):
    """Pin every RNG source so repeated runs stay the same."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


lock_every_seed(42)

IMAGE_SIZE  = 224
BATCH_SIZE  = 64
EPOCHS      = 200
MAX_LR      = 3e-3
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MIXUP_ALPHA = 0.3

print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


def locate_dataset_root():
    return pathlib.Path(
        "/kaggle/input/datasets/huynhthethien/radarcommunsignaldata2026train"
    )


def split_image_folder(dataset_root):
    complete_dataset = datasets.ImageFolder(root=str(dataset_root))
    category_names   = complete_dataset.classes
    category_count   = len(category_names)

    print(f"Total images  : {len(complete_dataset)}")
    print(f"Num classes   : {category_count}")
    print(f"Classes       : {category_names}")

    train_count      = int(0.95 * len(complete_dataset))
    validation_count = len(complete_dataset) - train_count
    seeded_generator = torch.Generator().manual_seed(42)
    sampled_parts    = random_split(
        complete_dataset,
        [train_count, validation_count],
        generator=seeded_generator,
    )

    print(f"Train samples : {train_count}")
    print(f"Val samples   : {validation_count}")

    return complete_dataset, category_count, category_names, sampled_parts


class SplitView(Dataset):
    """Attach split-specific transforms on top of a Subset."""

    def __init__(self, split_source, image_transform=None):
        self._split_source    = split_source
        self._image_transform = image_transform

    def __len__(self):
        return len(self._split_source)

    def __getitem__(self, item_index):
        image, target = self._split_source[item_index]
        if self._image_transform:
            image = self._image_transform(image)
        return image, target


def make_transforms():
    """Build train and validation transforms."""
    training_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), value=0),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    validation_pipeline = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return training_pipeline, validation_pipeline


def make_loaders(train_part, validation_part):
    train_augments, validation_augments = make_transforms()
    training_view   = SplitView(train_part,      image_transform=train_augments)
    validation_view = SplitView(validation_part, image_transform=validation_augments)

    training_batches   = DataLoader(
        training_view,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    validation_batches = DataLoader(
        validation_view,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return training_view, validation_view, training_batches, validation_batches


def mixup_batch(images, labels, num_classes, alpha=0.3):
    """
    Apply Mixup to a mini-batch.
    Returns mixed images and soft label pairs (lbl_a, lbl_b, lam).
    """
    if alpha <= 0:
        one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        return images, one_hot, one_hot, 1.0

    lam      = np.random.beta(alpha, alpha)
    lam      = max(lam, 1 - lam)
    batch_sz = images.size(0)
    perm     = torch.randperm(batch_sz, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[perm]

    one_hot_a = torch.zeros(batch_sz, num_classes, device=labels.device)
    one_hot_a.scatter_(1, labels.unsqueeze(1), 1.0)
    one_hot_b = one_hot_a[perm]

    return mixed_images, one_hot_a, one_hot_b, lam


class ChannelScaler(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channel_count: int, shrink_factor: int = 4):
        super().__init__()
        reduced_width     = max(channel_count // shrink_factor, 8)
        self.summary_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce_fc    = nn.Linear(channel_count, reduced_width,  bias=False)
        self.activate     = nn.ReLU(inplace=True)
        self.expand_fc    = nn.Linear(reduced_width,  channel_count, bias=False)
        self.gate         = nn.Sigmoid()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = feature_map.shape
        squeezed   = self.summary_pool(feature_map).view(b, c)
        w = self.gate(self.expand_fc(self.activate(self.reduce_fc(squeezed))))
        return feature_map * w.view(b, c, 1, 1)


class ResidualMobileUnit(nn.Module):
    """
    Inverted-residual block (MobileNetV2 style) + optional SE attention.
    stride=1 and matching channel count -> identity skip path.
    """

    def __init__(
        self,
        input_channels:      int,
        output_channels:     int,
        stride:              int,
        expansion_ratio:     int,
        apply_attention:     bool = True,
        attention_reduction: int  = 4,
    ):
        super().__init__()
        expanded      = round(input_channels * expansion_ratio)
        self.has_skip = stride == 1 and input_channels == output_channels

        layers = []
        if expansion_ratio != 1:
            layers += [
                nn.Conv2d(input_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1,
                      groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expanded, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        ]

        self.path              = nn.Sequential(*layers)
        self.channel_attention = (
            ChannelScaler(output_channels, attention_reduction)
            if apply_attention else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(self.path(x))
        return x + out if self.has_skip else out


class MultiScaleFusion(nn.Module):
    """
    Fuse mid-level and deep feature maps via learned 1x1 projections.
    Both tensors are global-average-pooled before addition so spatial
    sizes need not match.
    """

    def __init__(self, ch_early: int, ch_late: int, ch_out: int):
        super().__init__()
        self.proj_early = nn.Conv2d(ch_early, ch_out, 1, bias=False)
        self.proj_late  = nn.Conv2d(ch_late,  ch_out, 1, bias=False)
        self.bn         = nn.BatchNorm2d(ch_out)
        self.act        = nn.ReLU6(inplace=True)
        self.pool       = nn.AdaptiveAvgPool2d(1)

    def forward(self, early: torch.Tensor, late: torch.Tensor) -> torch.Tensor:
        early_gap = self.pool(early)
        late_gap  = self.pool(late)
        fused = self.act(self.bn(self.proj_early(early_gap) + self.proj_late(late_gap)))
        return fused


class SignalBackbone(nn.Module):
    """Compact CNN for RF signal spectrogram classification."""

    def __init__(self, num_categories: int = 12):
        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualMobileUnit(16, 16, stride=1, expansion_ratio=1, apply_attention=False),
            ResidualMobileUnit(16, 24, stride=2, expansion_ratio=6, apply_attention=True),
            ResidualMobileUnit(24, 24, stride=1, expansion_ratio=6, apply_attention=True),
        )

        self.stage2 = nn.Sequential(
            ResidualMobileUnit(24, 32, stride=2, expansion_ratio=6, apply_attention=True),
            ResidualMobileUnit(32, 32, stride=1, expansion_ratio=6, apply_attention=True),
        )

        self.stage3 = nn.Sequential(
            ResidualMobileUnit(32, 48, stride=2, expansion_ratio=5, apply_attention=True),
            ResidualMobileUnit(48, 48, stride=1, expansion_ratio=3, apply_attention=True),
            ResidualMobileUnit(48, 56, stride=2, expansion_ratio=3, apply_attention=True),
        )

        self.fusion = MultiScaleFusion(ch_early=32, ch_late=56, ch_out=56)

        self.regularizer  = nn.Dropout(0.40)
        self.output_layer = nn.Linear(56, num_categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x     = self.entry(x)
        x     = self.stage1(x)
        mid   = self.stage2(x)
        deep  = self.stage3(mid)
        fused = self.fusion(mid, deep)
        flat  = fused.view(fused.size(0), -1)
        flat  = self.regularizer(flat)
        return self.output_layer(flat)


BasicCNN           = SignalBackbone
SEBlock            = ChannelScaler
InvertedResidualSE = ResidualMobileUnit


def show_parameter_budget(network):
    trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"\n[INFO] Total trainable parameters: {trainable:,}")
    assert trainable < 100_000, (
        f"Parameter budget exceeded: {trainable:,} >= 100,000"
    )
    print(f"[INFO] Parameter constraint satisfied: {trainable:,} < 100,000 ✓")


def build_training_state(network, batch_stream, num_classes):
    """Create loss, optimizer, scheduler, and metric history."""
    loss_function     = nn.CrossEntropyLoss()
    parameter_updater = optim.AdamW(
        network.parameters(), lr=MAX_LR, weight_decay=0.02
    )
    step_schedule = optim.lr_scheduler.OneCycleLR(
        parameter_updater,
        max_lr=MAX_LR,
        steps_per_epoch=len(batch_stream),
        epochs=EPOCHS,
        pct_start=0.12,
        div_factor=10,
        final_div_factor=500,
        anneal_strategy="cos",
    )
    tracked_history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    return loss_function, parameter_updater, step_schedule, tracked_history


def move_batch(mini_batch):
    imgs, labels = mini_batch
    return imgs.to(DEVICE), labels.to(DEVICE)


def append_predictions(preds, targets, logits, labels):
    _, chosen = torch.max(logits, 1)
    preds.extend(chosen.cpu().numpy())
    targets.extend(labels.cpu().numpy())


def run_training_pass(
    network, batch_stream, loss_fn, opt, sched, num_classes
):
    network.train()
    cum_loss = 0.0
    epoch_preds, epoch_tgts = [], []

    for mini_batch in batch_stream:
        images, labels = move_batch(mini_batch)

        mixed_imgs, lbl_a, lbl_b, lam = mixup_batch(
            images, labels, num_classes, alpha=MIXUP_ALPHA
        )

        opt.zero_grad()
        logits    = network(mixed_imgs)
        log_probs = torch.log_softmax(logits, dim=1)
        loss_val  = -(
            lam       * (lbl_a * log_probs).sum(1) +
            (1 - lam) * (lbl_b * log_probs).sum(1)
        ).mean()

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        cum_loss += loss_val.item()
        append_predictions(epoch_preds, epoch_tgts, logits, labels)

    avg_loss = cum_loss / len(batch_stream)
    avg_acc  = accuracy_score(epoch_tgts, epoch_preds)
    return avg_loss, avg_acc, epoch_preds, epoch_tgts


def run_validation_pass(network, batch_stream, loss_fn):
    network.eval()
    cum_loss = 0.0
    epoch_preds, epoch_tgts = [], []

    with torch.no_grad():
        for mini_batch in batch_stream:
            images, labels = move_batch(mini_batch)
            logits         = network(images)
            loss_val       = loss_fn(logits, labels)
            cum_loss      += loss_val.item()
            append_predictions(epoch_preds, epoch_tgts, logits, labels)

    avg_loss = cum_loss / len(batch_stream)
    avg_acc  = accuracy_score(epoch_tgts, epoch_preds)
    return avg_loss, avg_acc, epoch_preds, epoch_tgts


def save_epoch_history(store, tl, vl, ta, va):
    store["train_loss"].append(tl)
    store["val_loss"].append(vl)
    store["train_acc"].append(ta)
    store["val_acc"].append(va)


def execute_training_loop(
    network, training_batches, validation_batches, loss_fn, num_classes
):
    best_val_acc      = 0.0
    stored_preds      = []
    stored_tgts       = []
    strongest_weights = None
    PATIENCE          = 30
    epochs_no_improve = 0

    print("\n[INFO] Starting training...")
    print("-" * 78)

    for ep in range(EPOCHS):
        tr_loss, tr_acc, _, _ = run_training_pass(
            network, training_batches, loss_fn, optimizer, scheduler, num_classes
        )
        vl_loss, vl_acc, val_preds, val_tgts = run_validation_pass(
            network, validation_batches, loss_fn
        )

        save_epoch_history(history, tr_loss, vl_loss, tr_acc, vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc      = vl_acc
            stored_preds      = val_preds
            stored_tgts       = val_tgts
            strongest_weights = copy.deepcopy(network.state_dict())
            epochs_no_improve = 0
            log_best_epoch(
                epoch_index=ep + 1,
                train_loss=tr_loss,
                train_acc=tr_acc,
                val_loss=vl_loss,
                val_acc=vl_acc,
                val_predictions=val_preds,
                val_targets=val_tgts,
                label_names=globals().get("class_names"),
            )
        else:
            epochs_no_improve += 1

        print(
            f"Epoch [{ep+1:03d}/{EPOCHS}] | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f} | "
            f"Best Val: {best_val_acc:.4f}"
        )

        if epochs_no_improve >= PATIENCE:
            print(f"\n[INFO] Early stopping triggered at epoch {ep+1}.")
            break

    if strongest_weights is not None:
        network.load_state_dict(strongest_weights)
        print(f"\n[INFO] Restored best checkpoint -> Val Acc = {best_val_acc:.4f}")

    return stored_tgts, stored_preds


def append_best_epoch_log(
    epoch_index,
    val_acc,
    precisions,
    recalls,
    f1_scores,
    supports,
    label_names,
    log_path="best_epoch_log.txt",
):
    labels = label_names or [f"class_{idx}" for idx in range(len(precisions))]

    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write("=" * 78 + "\n")
        handle.write(f"Epoch: {epoch_index}\n")
        handle.write(f"Val Accuracy: {val_acc:.4f}\n")
        header = (f"{'Class':<12}  {'Precision':>10}  {'Recall':>10}  "
                  f"{'F1-score':>10}  {'Support':>8}")
        handle.write(header + "\n")
        handle.write("-" * len(header) + "\n")
        for idx, name in enumerate(labels):
            handle.write(
                f"{name:<12}  {precisions[idx]:>10.4f}  {recalls[idx]:>10.4f}  "
                f"{f1_scores[idx]:>10.4f}  {int(supports[idx]):>8}\n"
            )
        handle.write("\n")


def log_best_epoch(
    epoch_index,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    val_predictions,
    val_targets,
    label_names=None,
):
    metric_labels = list(range(len(label_names))) if label_names is not None else None
    prec, rec, f1, sup = precision_recall_fscore_support(
        val_targets,
        val_predictions,
        labels=metric_labels,
        zero_division=0,
    )

    print("\n" + "=" * 78)
    print("[INFO] NEW BEST VALIDATION ACCURACY")
    print("=" * 78)
    print(f"Epoch      : {epoch_index}")
    print(f"Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}")
    print(f"Val Loss   : {val_loss:.4f} | Val Acc   : {val_acc:.4f}")
    print("-" * 78)

    header = (f"{'Class':<12}  {'Precision':>10}  {'Recall':>10}  "
              f"{'F1-score':>10}  {'Support':>8}")
    print(header)
    print("-" * len(header))

    labels = label_names or [f"class_{idx}" for idx in range(len(prec))]
    for idx, name in enumerate(labels):
        print(f"{name:<12}  {prec[idx]:>10.4f}  {rec[idx]:>10.4f}  "
              f"{f1[idx]:>10.4f}  {int(sup[idx]):>8}")
    print("=" * 78)

    append_best_epoch_log(
        epoch_index=epoch_index,
        val_acc=val_acc,
        precisions=prec,
        recalls=rec,
        f1_scores=f1,
        supports=sup,
        label_names=label_names,
    )


def print_class_metrics(targets, predictions, label_names):
    print("\n" + "=" * 78)
    print("[INFO] DETAILED CLASSIFICATION REPORT")
    print("=" * 78)

    overall_acc = accuracy_score(targets, predictions)
    prec, rec, f1, sup = precision_recall_fscore_support(
        targets, predictions, zero_division=0
    )

    print(f"Overall Accuracy: {overall_acc:.4f}\n")
    hdr = (f"{'Class':<12}  {'Precision':>10}  {'Recall':>10}  "
           f"{'F1-score':>10}  {'Support':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(label_names):
        print(f"{name:<12}  {prec[i]:>10.4f}  {rec[i]:>10.4f}  "
              f"{f1[i]:>10.4f}  {int(sup[i]):>8}")
    print("=" * 78)


def save_loss_chart(store):
    ep = range(1, len(store["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(ep, store["train_loss"], label="Training Loss",   linewidth=2)
    plt.plot(ep, store["val_loss"],   label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=13); plt.ylabel("Loss", fontsize=13)
    plt.title("Training and Validation Loss", fontsize=15)
    plt.legend(fontsize=12); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300); plt.close()


def save_accuracy_chart(store):
    ep = range(1, len(store["train_acc"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(ep, store["train_acc"], label="Training Accuracy",   linewidth=2)
    plt.plot(ep, store["val_acc"],   label="Validation Accuracy", linewidth=2)
    plt.axhline(y=0.90, color="red", linestyle="--", alpha=0.7, label="90% target")
    plt.xlabel("Epoch", fontsize=13); plt.ylabel("Accuracy", fontsize=13)
    plt.title("Training and Validation Accuracy", fontsize=15)
    plt.legend(fontsize=12); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300); plt.close()


def save_confusion_matrix_chart(targets, predictions, label_names):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label",      fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300); plt.close()
    print("[INFO] Saved all plots successfully.")


dataset_root = locate_dataset_root()
base_dataset, num_classes, class_names, subset_pair = split_image_folder(dataset_root)
train_subset, val_subset = subset_pair

train_dataset, val_dataset, train_loader, val_loader = make_loaders(
    train_subset, val_subset
)

model = SignalBackbone(num_classes).to(DEVICE)
show_parameter_budget(model)

criterion, optimizer, scheduler, history = build_training_state(
    model, train_loader, num_classes
)

final_labels, final_preds = execute_training_loop(
    model, train_loader, val_loader, criterion, num_classes
)

print_class_metrics(final_labels, final_preds, class_names)
save_loss_chart(history)
save_accuracy_chart(history)
train_targets = []
train_preds = []

model.eval()
with torch.no_grad():
    for batch in train_loader:
        images, labels = move_batch(batch)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        train_preds.extend(predicted.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())

save_confusion_matrix_chart(train_targets, train_preds, class_names)


############################################
# DO NOT MODIFY THIS SECTION
############################################
model.eval()
example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
traced_model  = torch.jit.trace(model, example_input)
GroupID       = "03"
model_name    = f"{GroupID}_DeepLearning Project_TrainedModel.pt"
traced_model.save(model_name)
print("Model saved:", model_name)
