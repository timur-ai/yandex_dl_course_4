"""Configuration for paths, hyperparameters, backbone, and runtime flags."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable configuration for the project.

    Attributes:
        project_dir: Repository root directory.
        data_dir: Directory containing dataset files.
        images_dir: Directory with per-dish image folders.
        dishes_csv: Path to the dishes CSV file.
        ingredients_csv: Path to the ingredients CSV file.
        models_dir: Directory for model checkpoints.
        outputs_dir: Directory for run outputs.
        outputs_figs_dir: Directory for EDA/analysis figures.
        outputs_preds_image_csv: Predictions CSV for image baseline.
        outputs_preds_tfidf_csv: Predictions CSV for TF-IDF baseline.
        outputs_preds_fusion_csv: Predictions CSV for fusion model.
        best_image_ckpt: Best checkpoint path for image baseline.
        best_ckpt: Best checkpoint path for fusion model.
        seed: Global RNG seed.
        device_preference: Preferred device selection policy.
        use_amp: Enable mixed precision when CUDA is available.
        num_workers: DataLoader worker processes.
        pin_memory: Enable pinned memory for CUDA transfer.
        persistent_workers: Keep DataLoader workers alive between epochs.
        batch_size: Mini-batch size.
        epochs: Training epochs.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        grad_clip_norm: Max norm for gradient clipping; disabled if None.
        compile: Enable torch.compile when supported.
        backbone: timm backbone name.
        image_size: Square resize size used in transforms.
        tfidf_min_df: Minimum document frequency for TF-IDF.
        tfidf_max_df: Maximum document frequency for TF-IDF.
        tfidf_max_features: Cap on TF-IDF vocabulary size.
        tfidf_norm: Normalization used in TF-IDF.
        tfidf_use_idf: Whether to enable IDF reweighting.
        tfidf_sublinear_tf: Whether to use sublinear TF scaling.
    """

    project_dir: Path = Path(__file__).resolve().parents[1]

    # Paths
    data_dir: Path = project_dir / "data"
    images_dir: Path = data_dir / "images"
    dishes_csv: Path = data_dir / "dish.csv"
    ingredients_csv: Path = data_dir / "ingredients.csv"
    models_dir: Path = project_dir / "models"
    outputs_dir: Path = project_dir / "outputs"
    outputs_figs_dir: Path = outputs_dir / "figs"

    # Outputs
    outputs_preds_image_csv: Path = outputs_dir / "preds_image.csv"
    outputs_preds_tfidf_csv: Path = outputs_dir / "preds_tfidf.csv"
    outputs_preds_fusion_csv: Path = outputs_dir / "preds.csv"
    best_image_ckpt: Path = models_dir / "best_image.pt"
    best_ckpt: Path = models_dir / "best.pt"

    # Runtime
    seed: int = 42
    device_preference: Literal["cuda", "cpu", "auto"] = "auto"
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    batch_size: int = 24
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = None
    compile: bool = False

    # Development mode (quick passes for correctness checks)
    dev_mode: bool = False
    dev_epochs: int = 1
    dev_max_train_batches: int = 4
    dev_max_val_batches: int = 4

    # Model
    backbone: str = "tf_efficientnet_b3"
    image_size: int = 300

    # TF-IDF
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    tfidf_max_features: int = 10000
    tfidf_norm: Literal["l2", "l1"] = "l2"
    tfidf_use_idf: bool = True
    tfidf_sublinear_tf: bool = True


