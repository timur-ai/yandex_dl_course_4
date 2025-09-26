"""Models, training/evaluation utilities, metrics, and checkpointing."""
from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm  # adaptive tqdm: picks notebook widget or console automatically


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for determinism.

    Args:
        seed: RNG seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*paths: Path | str) -> None:
    """Create directories if they do not exist.

    Args:
        *paths: Paths to create.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


class ImageBackbone(nn.Module):
    """timm backbone with pooled feature output.

    Creates a timm model with num_classes=0 to expose feature extractor with
    global pooling applied. Forward returns a 2D tensor of shape [N, C].
    """

    def __init__(self, backbone_name: str) -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone_name, pretrained=True, num_classes=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class ImageRegressor(nn.Module):
    """MLP regressor head on top of frozen or trainable backbone features."""

    def __init__(self, in_features: int, hidden_features: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)


class FusionRegressor(nn.Module):
    """Projects image and tfidf features to a common dim and regresses.

    image_features -> Linear(image_in, proj_dim)
    tfidf_features -> Linear(tfidf_in, proj_dim)
    concat -> MLP -> scalar
    """

    def __init__(
        self,
        image_in_features: int,
        tfidf_in_features: int,
        projection_dim: int = 512,
        hidden_features: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_in_features, projection_dim)
        self.tfidf_proj = nn.Linear(tfidf_in_features, projection_dim)
        self.regressor = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, image_feats: torch.Tensor, tfidf_feats: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        img = self.image_proj(image_feats)
        txt = self.tfidf_proj(tfidf_feats)
        fused = torch.cat([img, txt], dim=1)
        return self.regressor(fused).squeeze(-1)

@torch.no_grad()
def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_one_epoch(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool = False,
    max_grad_norm: float | None = None,
    max_batches: int | None = None,
    fusion: bool = False,
) -> float:
    model.train()
    feature_extractor.train()
    amp_enabled = bool(amp_enabled and (device.type == "cuda"))
    loss_sum = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc="train_fusion" if fusion else "train",
            leave=True,
            dynamic_ncols=True,
        )
    ):
        if fusion:
            images, targets, tfidf_vecs, _ = batch
        else:
            images, targets, _ = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            img_feats = feature_extractor(images)
            if fusion:
                tfidf_vecs_t = torch.as_tensor(
                    tfidf_vecs, dtype=torch.float32, device=device
                )
                preds = model(img_feats, tfidf_vecs_t)
            else:
                preds = model(img_feats)
            loss = criterion(preds, targets)
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.item()) * images.size(0)
        if (max_batches is not None) and (batch_idx + 1 >= max_batches):
            break
    return loss_sum / len(dataloader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    fusion: bool = False,
) -> tuple[float, dict[str, float]]:
    model.eval()
    feature_extractor.eval()
    loss_sum = 0.0
    all_true: list[float] = []
    all_pred: list[float] = []
    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc="validate_fusion" if fusion else "validate",
            leave=True,
            dynamic_ncols=True,
        )
    ):
        if fusion:
            images, targets, tfidf_vecs, _ = batch
        else:
            images, targets, _ = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device)
        img_feats = feature_extractor(images)
        if fusion:
            tfidf_vecs_t = torch.as_tensor(
                tfidf_vecs, dtype=torch.float32, device=device
            )
            preds = model(img_feats, tfidf_vecs_t)
        else:
            preds = model(img_feats)
        loss = criterion(preds, targets)
        loss_sum += float(loss.item()) * images.size(0)
        all_true.extend(targets.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())
        if (max_batches is not None) and (batch_idx + 1 >= max_batches):
            break
    val_loss = loss_sum / len(dataloader.dataset)
    metrics = evaluate_metrics(np.asarray(all_true), np.asarray(all_pred))
    return val_loss, metrics


def save_best_checkpoint(
    model_state: dict,
    optimizer_state: dict,
    epoch: int,
    best_path: Path,
) -> None:
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model_state, "optimizer": optimizer_state}, best_path)


def train_one_epoch_fusion(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool = False,
    max_grad_norm: float | None = None,
    max_batches: int | None = None,
) -> float:
    return train_one_epoch(
        model,
        feature_extractor,
        dataloader,
        criterion,
        optimizer,
        device,
        amp_enabled,
        max_grad_norm,
        max_batches,
        True,
    )


@torch.no_grad()
def validate_fusion(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float]]:
    return validate(
        model,
        feature_extractor,
        dataloader,
        criterion,
        device,
        max_batches,
        True,
    )


