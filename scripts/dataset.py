"""Datasets and image/text transform builders."""
from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from timm import create_model
from timm.data import resolve_model_data_config
from timm.data.transforms_factory import create_transform


def build_train_transforms(backbone: str, image_size: int) -> Any:
    """Return training transforms using timm pretrained config.

    Args:
        backbone: timm backbone name.
        image_size: square side for resize/crop.
    """
    model = create_model(backbone, pretrained=True, num_classes=0)
    cfg = resolve_model_data_config(model)
    cfg["input_size"] = (3, image_size, image_size)
    return create_transform(**cfg, is_training=True)


def build_eval_transforms(backbone: str, image_size: int) -> Any:
    """Return evaluation transforms using timm pretrained config.

    Args:
        backbone: timm backbone name.
        image_size: square side for resize/crop.
    """
    model = create_model(backbone, pretrained=True, num_classes=0)
    cfg = resolve_model_data_config(model)
    cfg["input_size"] = (3, image_size, image_size)
    return create_transform(**cfg, is_training=False)


class ImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Dataset returning (image, target, dish_id).

    Expects images at: images_dir/<dish_id>/rgb.png

    Args:
        df: DataFrame with at least dish and target columns.
        images_dir: Root directory with per-dish folders.
        transform: Transform callable (e.g., from timm) applied to PIL Image.
        dish_id_col: Column name for dish id.
        target_col: Column name for regression target.
        image_name: Image filename within each dish directory.

    Returns per item:
        image: Tensor [3, H, W]
        target: Scalar float32 tensor
        dish_id: str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform: Any,
        dish_id_col: str = "dish_id",
        target_col: str = "total_calories",
        image_name: str = "rgb.png",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.dish_id_col = dish_id_col
        self.target_col = target_col
        self.image_name = image_name

    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[index]
        dish_id = str(row[self.dish_id_col])
        target = float(row[self.target_col])
        img_path = self.images_dir / str(dish_id) / self.image_name
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return image, target_tensor, dish_id


class FusionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, np.ndarray, str]]):
    """Dataset returning (image, target, tfidf_features, dish_id).

    TF-IDF features are retrieved via id-to-row mapping from a matrix.

    Args:
        df: DataFrame with dish and target columns.
        images_dir: Root directory with per-dish folders.
        transform: Transform callable applied to PIL Image.
        tfidf_matrix: Row-indexable matrix (supports .getrow for CSR or ndarray indexing).
        id_to_row: Mapping from dish_id to row index in TF-IDF matrix.
        dish_id_col: Column name for dish id.
        target_col: Column name for regression target.
        image_name: Image filename within each dish directory.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform: Any,
        tfidf_matrix: Any,
        id_to_row: dict[str, int],
        dish_id_col: str = "dish_id",
        target_col: str = "total_calories",
        image_name: str = "rgb.png",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.tfidf_matrix = tfidf_matrix
        self.id_to_row = id_to_row
        self.dish_id_col = dish_id_col
        self.target_col = target_col
        self.image_name = image_name

    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, str]:
        row = self.df.iloc[index]
        dish_id = str(row[self.dish_id_col])
        target = float(row[self.target_col])
        img_path = self.images_dir / str(dish_id) / self.image_name
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Fetch TF-IDF row by dish_id
        row_idx = self.id_to_row[dish_id]
        getrow = getattr(self.tfidf_matrix, "getrow", None)
        if getrow is not None:
            tfidf_vec = getrow(row_idx).toarray().ravel()
        else:
            tfidf_vec = np.asarray(self.tfidf_matrix[row_idx]).ravel()
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return image, target_tensor, tfidf_vec, dish_id


# Mass-aware variants returning total_mass alongside features
class ImageDatasetMass(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform: Any,
        dish_id_col: str = "dish_id",
        target_col: str = "total_calories",
        mass_col: str = "total_mass",
        image_name: str = "rgb.png",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.dish_id_col = dish_id_col
        self.target_col = target_col
        self.mass_col = mass_col
        self.image_name = image_name

    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[index]
        dish_id = str(row[self.dish_id_col])
        target = float(row[self.target_col])
        mass = float(row[self.mass_col])
        img_path = self.images_dir / str(dish_id) / self.image_name
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return (
            image,
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(mass, dtype=torch.float32),
            dish_id,
        )


class FusionDatasetMass(
    Dataset[tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor, str]]
):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform: Any,
        tfidf_matrix: Any,
        id_to_row: dict[str, int],
        dish_id_col: str = "dish_id",
        target_col: str = "total_calories",
        mass_col: str = "total_mass",
        image_name: str = "rgb.png",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.tfidf_matrix = tfidf_matrix
        self.id_to_row = id_to_row
        self.dish_id_col = dish_id_col
        self.target_col = target_col
        self.mass_col = mass_col
        self.image_name = image_name

    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor, str]:
        row = self.df.iloc[index]
        dish_id = str(row[self.dish_id_col])
        target = float(row[self.target_col])
        mass = float(row[self.mass_col])
        img_path = self.images_dir / str(dish_id) / self.image_name
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        row_idx = self.id_to_row[dish_id]
        getrow = getattr(self.tfidf_matrix, "getrow", None)
        if getrow is not None:
            tfidf_vec = getrow(row_idx).toarray().ravel()
        else:
            tfidf_vec = np.asarray(self.tfidf_matrix[row_idx]).ravel()
        return (
            image,
            torch.tensor(target, dtype=torch.float32),
            tfidf_vec,
            torch.tensor(mass, dtype=torch.float32),
            dish_id,
        )

