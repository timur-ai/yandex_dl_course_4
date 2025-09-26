## Yandex DL Course 4 — Calories Estimation from Image and Ingredients

End-to-end project that predicts a dish's total calories from a plate image and its ingredients. It includes:

- Image-only baseline using a timm backbone + MLP regressor
- Text-only baseline with TF‑IDF over ingredients
- Fusion model combining image features with TF‑IDF features

Target metric: mean absolute error (MAE) < 50.


### Features
- Pretrained image backbones via `timm`
- Clean `torch` training/validation utilities with AMP and gradient clipping
- `Dataset` implementations for image-only and fusion setups
- Configuration in a single `Config` dataclass
- Notebook to reproduce baselines and save predictions/checkpoints


### Project layout
- `scripts/`
  - `config.py` — global paths, hyperparameters, and runtime flags
  - `dataset.py` — `ImageDataset`, `FusionDataset`, and transform builders
  - `utils.py` — backbone/regressor modules, train/validate loops, metrics, ckpt I/O
- `data/`
  - `dish.csv` — rows with `dish_id,total_calories,total_mass,ingredients,split`
  - `ingredients.csv` — mapping of ingredient ids to names (`id,ingr`)
  - `images/<dish_id>/rgb.png` — RGB image for each dish
- `notebook.ipynb` — EDA, TF‑IDF pipeline, training and predictions
- `models/` — saved checkpoints, e.g. `best_image.pt`, `best.pt` (fusion)
- `outputs/` — figures and predictions (`preds_image.csv`, `preds_tfidf.csv`, `preds.csv`)
- `pyproject.toml` — dependencies and metadata


### Dataset layout
- `data/dish.csv` columns:
  - `dish_id` (e.g., `dish_1561662216`)
  - `total_calories` (float, regression target)
  - `total_mass` (float)
  - `ingredients` (semicolon-separated tokens, e.g., `ingr_0000000312;...`)
  - `split` (`train`/`test`)
- `data/ingredients.csv` has two columns: `id,ingr` (ingredient id → name)
- Images are stored under `data/images/<dish_id>/rgb.png`


### Requirements
- Python >= 3.10
- Recommended: CUDA GPU for training; CPU works but is slow

Install dependencies (uv):

```bash
uv add torch timm scikit-learn pandas numpy pillow matplotlib notebook ipywidgets tqdm
```


### Getting the data
The dataset is hosted on Yandex.Disk (public link is in `download_dataset.txt`). You can:

- Download manually via the public link in `download_dataset.txt` and extract into `data/` so that you have `data/dish.csv`, `data/ingredients.csv`, and `data/images/...`.
- Or use Bash (Linux/macOS/WSL) with `curl`, `jq`, `wget`, `unzip`:

```bash
url="https://disk.yandex.ru/d/kz9g5msVqtahiw"
direct_url=$(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=$url" | jq -r '.href')
wget -O ogyeiv2.zip "$direct_url"
unzip -qq ogyeiv2.zip -d data
```

On Windows without WSL, prefer manual download or use PowerShell to download the archive, then unzip to `data/`.


### Quickstart
1) Verify the dataset layout as above.
2) Open and run the notebook to reproduce baselines and create outputs/checkpoints:

```bash
uv run jupyter notebook
# open notebook.ipynb and run all cells
```

3) Or run a minimal image-baseline training loop from Python (single epoch example):

```python
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from scripts.config import Config
from scripts.dataset import ImageDataset, build_train_transforms, build_eval_transforms
from scripts.utils import ImageBackbone, ImageRegressor, seed_everything, train_one_epoch, validate

cfg = Config()
seed_everything(cfg.seed)

df = pd.read_csv(cfg.dishes_csv)
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "test"].reset_index(drop=True)

train_tfms = build_train_transforms(cfg.backbone, cfg.image_size)
val_tfms = build_eval_transforms(cfg.backbone, cfg.image_size)

train_ds = ImageDataset(train_df, cfg.images_dir, train_tfms)
val_ds = ImageDataset(val_df, cfg.images_dir, val_tfms)

train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                      pin_memory=cfg.pin_memory, persistent_workers=cfg.persistent_workers)
val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory, persistent_workers=cfg.persistent_workers)

device = torch.device("cuda" if (cfg.device_preference != "cpu" and torch.cuda.is_available()) else "cpu")
feature_extractor = ImageBackbone(cfg.backbone).to(device)

# Infer backbone output size by a forward pass of one batch
with torch.no_grad():
    x0, _, _ = next(iter(val_dl))
    x0 = x0.to(device)
    feat_dim = feature_extractor(x0).shape[1]

model = ImageRegressor(in_features=feat_dim).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
crit = torch.nn.L1Loss()

train_loss = train_one_epoch(model, feature_extractor, train_dl, crit, opt, device,
                             amp_enabled=cfg.use_amp, max_grad_norm=cfg.grad_clip_norm,
                             max_batches=(cfg.dev_max_train_batches if cfg.dev_mode else None))
val_loss, metrics = validate(model, feature_extractor, val_dl, crit, device,
                             max_batches=(cfg.dev_max_val_batches if cfg.dev_mode else None))

print({"train_loss": train_loss, "val_loss": val_loss, **metrics})
```


### Fusion model (image + TF‑IDF)
The notebook demonstrates building a TF‑IDF matrix from `ingredients` and then using `FusionDataset` with `FusionRegressor`:

1) Fit a `TfidfVectorizer` on ingredient tokens and persist the vectorizer and sparse matrix (see `outputs/tfidf_vectorizer.pkl`).
2) Build `id_to_row` mapping from `dish_id` to TF‑IDF row index.
3) Use `FusionDataset` to feed `(image, target, tfidf_vector, dish_id)` to the model.

Note: Mapping from tokens like `ingr_0000000312` to human-readable names is optional for the baseline; TF‑IDF can operate directly on the tokens if consistent.


### Configuration
All runtime parameters live in `scripts/config.py` (`Config` dataclass):

- Paths: `data_dir`, `images_dir`, `dishes_csv`, `ingredients_csv`, `models_dir`, `outputs_dir`
- Outputs: `outputs_preds_image_csv`, `outputs_preds_tfidf_csv`, `outputs_preds_fusion_csv`, `best_image_ckpt`, `best_ckpt`
- Runtime: `seed`, `device_preference` ("auto"/"cuda"/"cpu"), `use_amp`, loader workers, `batch_size`, `epochs`, LR, weight decay, `grad_clip_norm`, `compile`
- Dev mode: `dev_mode` with small `dev_epochs`, and caps for `max_*_batches` to smoke-test quickly
- Model: `backbone` (e.g., `tf_efficientnet_b0`), `image_size`
- TF‑IDF: `min_df`, `max_df`, `max_features`, `norm`, `use_idf`, `sublinear_tf`


### Reproducing outputs
- Image baseline predictions → `outputs/preds_image.csv`
- TF‑IDF baseline predictions → `outputs/preds_tfidf.csv`
- Fusion predictions → `outputs/preds.csv`

Use the notebook to generate these files and the best checkpoints:

- `models/best_image.pt` — best image-only regressor head
- `models/best.pt` — best fusion regressor


### Inference from checkpoints (image-only example)

```python
import torch
from scripts.config import Config
from scripts.utils import ImageBackbone, ImageRegressor

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = ImageBackbone(cfg.backbone).to(device).eval()
model = ImageRegressor(in_features=feature_extractor(torch.randn(1,3,cfg.image_size,cfg.image_size).to(device)).shape[1]).to(device)
state = torch.load(cfg.best_image_ckpt, map_location=device)
model.load_state_dict(state["model"])  # checkpoint format from save_best_checkpoint
model.eval()

# Prepare a single preprocessed image tensor `x` then do:
# y = model(feature_extractor(x.to(device))).item()
```


### Tips
- If training is too slow or you only want to verify wiring, set `dev_mode=True` in `Config`.
- Increase `num_workers` and enable `pin_memory` on CUDA for faster data loading.
- Set `device_preference` to `cpu` if you lack a GPU.


### License
For coursework use. If you plan to reuse beyond the course, check with the authors.


