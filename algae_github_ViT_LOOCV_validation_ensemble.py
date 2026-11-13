#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Physics-informed Vision Transformer (PViT) for dissolved-oxygen (DO) sensing
under algal biofouling -- validation-selected Leave-One-Day-Out CV (LOOCV)
==============================================================================

This script generates the validation-selected LOOCV results reported in the
manuscript (Fig. 8B; Table S2, "PViT (LOOCV + val, ensemble-3)").

Evaluation protocol (per fold; one fold per experimental day)
-------------------------------------------------------------
  * Test day       : one experimental day, held out entirely.
  * Validation day : the chronologically preceding day (for the first day,
                     the following day).
  * Training days  : all remaining days (11 of the 13 days).

  1. Three ensemble members (ENSEMBLE_SIZE) are trained on the training days
     only, each from a different random initialisation of the task heads.
  2. For each member, the checkpoint is the epoch with the lowest
     validation-day MAE (best-validation epoch).
  3. Only after steps 1-2 are complete is the test-day loader created and the
     test day evaluated, once. The test day is never used for checkpointing,
     member selection, early stopping or any other decision.

  Two test results are reported for every fold; neither uses the test day
  for any choice:
    (a) Ensemble mean               -- mean prediction of the three members
                                       (no choice between members); the
                                       ensemble standard deviation is saved as
                                       the uncertainty estimate.
    (b) Validation-selected member  -- the member with the lowest
                                       validation-day MAE.
  Overall results are given both as the mean of the per-day test MAEs and as
  the MAE pooled over all test frames.

Outputs (all inside --output-dir, default ./outputs/PViT_LOOCV_validation)
--------------------------------------------------------------------------
  features/pinn_features.parquet   extracted frames with DO and temperature
  hpo/best_hpo_params.json         tuned hyperparameters
  red_intensity_heatmaps/          mean red-intensity map per day
  fold_plan.csv                    test / validation day of every fold
  fold_cache/fold_XX/              weights, training history, test predictions
                                   and summary of every fold
  analysis/4_final_results/        final metrics, plots and the data behind them

Hyperparameters
---------------
  Four global hyperparameters (learning rate, physics-loss weight, embedding
  dimension, weight decay) are tuned once with Optuna before the LOOCV, on a
  30% frame subset pooled across days (random 80/20 frame split), and are then
  shared unchanged by every fold. No fold-specific tuning uses the test day.
  To use the hyperparameters of the reported run instead of re-tuning, pass
  them with --hyperparameters (a JSON file with keys lr, lambda_physics,
  embedding_dim, weight_decay).

Reproducibility
---------------
  Seeds are fixed, but cuDNN autotuning and mixed precision are enabled for
  speed, so repeated GPU runs agree closely but not bit-for-bit.

Usage
-----
  python algae_github_ViT_LOOCV_validation.py --data-root ./data/raw
  python algae_github_ViT_LOOCV_validation.py --data-root ./data/raw --folds 1,2

  Finished ensemble members and folds are cached as they complete, so an
  interrupted run continues where it stopped when the same command is run
  again. --folds runs a subset of folds (e.g. one process per GPU); the final
  report is written automatically once all folds are finished.

  Each experimental day is a sub-folder of --data-root whose name
  contains its date (DD-MM-YYYY or DD_MM_YYYY) and which contains:
      *_ROI.mp4           cropped sensor video (file name has YYYYMMDD_HHMMSS)
      *_arduino_*.txt     reference DO sensor log
      *temperature*.csv   temperature log
==============================================================================
"""

import argparse
import functools
import glob
import json
import os
import random
import re
import shutil
import sys
import traceback
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.stats
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use("seaborn-v0_8-whitegrid")

# ==============================================================================
# CONFIGURATION (values used for the reported result)
# ==============================================================================
# --- Default paths (override with --data-root / --output-dir) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(SCRIPT_DIR, "data", "raw")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "PViT_LOOCV_validation")

# --- Evaluation protocol ---
# Validation day = chronologically preceding day (following day for the first day);
# checkpoint = best-validation epoch of each member.

# --- Data processing ---
FRAME_SKIP = 15                      # 30 fps -> 2 fps
STORAGE_RESIZE_DIM = (48, 48)
MODEL_INPUT_DIM = (224, 224)
FORCED_VIDEO_FPS = 30
FRAME_CHUNK_SIZE = 2500
# Stable-DO windows used for analysis (seconds from start of each video)
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300), (3900, 4100), (4700, 4900)]
SENSOR_RESAMPLE_WINDOW_S = 10

# --- Hyperparameter optimisation ---
N_TRIALS_FOR_TUNING = 10
N_EPOCHS_FOR_TUNING = 15
HPO_DATA_SUBSET_FRACTION = 0.3

# --- Training ---
FORCE_RERUN_CV = False
N_EPOCHS_FOR_FINAL_TRAINING = 35
ENSEMBLE_SIZE = 3
BATCH_SIZE = 64
GRAD_ACCUMULATION_STEPS = 8
GRADIENT_CLIP_VALUE = 1.0
USE_LR_SCHEDULER = True              # cosine annealing over the full run
USE_CURRICULUM_LEARNING = True       # physics weight ramped up over first epochs
LAMBDA_CURRICULUM_START = 0.01
LAMBDA_CURRICULUM_EPOCHS = 5
# Frames are ImageNet-normalised; no flip augmentation (as in the reported runs).
USE_DATA_AUGMENTATION = False

# --- Model ---
PHYSICS_MODEL = "nonlinear"          # two-site Stern-Volmer
USE_CONFIDENCE_LOSS = True
USE_TRANSFER_LEARNING = True
TRANSFER_MODEL_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
DEVICE_TYPE = DEVICE.type

try:
    NUM_WORKERS = max(1, int(os.environ.get("NSLOTS", cpu_count())) - 1)
except (ValueError, TypeError):
    NUM_WORKERS = max(1, cpu_count() - 1)

IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ==============================================================================
# STAGE 1: DATA DISCOVERY AND PREPROCESSING
# ==============================================================================

def parse_arduino_log(file_path: str) -> pd.DataFrame:
    """Reference DO log -> regular time grid (linear interpolation over gaps)."""
    timestamps, oxygen_values = [], []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "MEA" not in line:
                    continue
                try:
                    parts = line.split()
                    dt_object = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
                    raw_oxygen_value = int(parts[parts.index("MEA") + 5])
                    timestamps.append(dt_object)
                    oxygen_values.append(raw_oxygen_value / 1000.0)
                except (ValueError, IndexError, TypeError):
                    pass
    except FileNotFoundError:
        return pd.DataFrame()

    if not timestamps:
        return pd.DataFrame()
    df = pd.DataFrame({"timestamp": timestamps, "oxygen_umol_L": oxygen_values})
    if df.timestamp.duplicated().any():
        df = df.groupby("timestamp", as_index=False)["oxygen_umol_L"].mean()
    df.set_index("timestamp", inplace=True)
    if SENSOR_RESAMPLE_WINDOW_S > 0:
        df = df.resample(f"{SENSOR_RESAMPLE_WINDOW_S}s").mean()
        df = df.interpolate(method="linear", limit_direction="both").dropna().reset_index()
    else:
        df = df.reset_index()
    return df


def parse_temperature_log(file_path: str) -> pd.DataFrame:
    """Temperature log -> spike removal (>100% step) -> regular time grid."""
    try:
        df = pd.read_csv(file_path)
        df.columns = ["timestamp", "temperature_C"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["temperature_C"] = pd.to_numeric(df["temperature_C"], errors="coerce")
        df.dropna(inplace=True)
        if df.empty:
            return pd.DataFrame()
        if df.timestamp.duplicated().any():
            df = df.groupby("timestamp", as_index=False)["temperature_C"].mean()
        df.sort_values("timestamp", inplace=True, ignore_index=True)
        spike_mask = df["temperature_C"].pct_change().abs() >= 1.0
        if spike_mask.sum() > 0:
            df = df[~spike_mask].copy()
        df.set_index("timestamp", inplace=True)
        if SENSOR_RESAMPLE_WINDOW_S > 0:
            df = df.resample(f"{SENSOR_RESAMPLE_WINDOW_S}s").mean()
            df = df.interpolate(method="linear", limit_direction="both").dropna().reset_index()
        else:
            df = df.reset_index()
        return df
    except Exception as e:
        print(f"Warning: could not parse temperature file {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()


def find_experiment_files(root_dir: str) -> list:
    print("--- Discovering experiment files ---")
    dated_dirs = []
    for d in glob.glob(os.path.join(root_dir, "*")):
        if not os.path.isdir(d):
            continue
        match = re.search(r"(\d{1,2}[-_]\d{2}[-_]\d{4})", os.path.basename(d))
        if match:
            try:
                date = datetime.strptime(match.group(1).replace("_", "-"), "%d-%m-%Y")
                dated_dirs.append({"path": d, "date": date})
            except ValueError:
                pass
    dated_dirs.sort(key=lambda x: x["date"])

    valid = []
    for entry in dated_dirs:
        exp_dir = entry["path"]
        videos = glob.glob(os.path.join(exp_dir, "**", "*_ROI.mp4"), recursive=True)
        arduino = glob.glob(os.path.join(exp_dir, "**", "*_arduino_*.txt"), recursive=True)
        temps = (glob.glob(os.path.join(exp_dir, "**", "*temperature*.csv"), recursive=True)
                 + glob.glob(os.path.join(exp_dir, "**", "*temperature*.txt"), recursive=True))
        if videos and arduino and temps:
            valid.append({"id": os.path.basename(exp_dir), "video_path": videos[0],
                          "raw_arduino_path": arduino[0], "temperature_path": temps[0],
                          "date": entry["date"]})
    if not valid:
        raise FileNotFoundError(f"No valid experiments found under {root_dir}. "
                                "Folder names must contain a date (DD-MM-YYYY or DD_MM_YYYY).")
    print(f"Found {len(valid)} valid experiments.")
    return valid


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    match = re.search(r"(\d{8}_\d{6})", filename)
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else None


def process_experiment_chunked(args: dict) -> bool:
    """Sample every FRAME_SKIP-th frame, attach nearest DO and temperature."""
    exp, frame_skip, resize_dim, temp_dir = args["exp"], args["frame_skip"], args["resize_dim"], args["temp_dir"]
    exp_id = exp["id"]
    try:
        df_arduino = parse_arduino_log(exp["raw_arduino_path"])
        df_temp = parse_temperature_log(exp["temperature_path"])
        if df_arduino.empty or df_temp.empty:
            return False
        df_arduino.set_index("timestamp", inplace=True)
        df_temp.set_index("timestamp", inplace=True)

        cap = cv2.VideoCapture(exp["video_path"])
        if not cap.isOpened():
            return False
        fps = FORCED_VIDEO_FPS if FORCED_VIDEO_FPS else cap.get(cv2.CAP_PROP_FPS)
        video_start_time = extract_timestamp_from_filename(os.path.basename(exp["video_path"]))
        if not fps or video_start_time is None:
            cap.release()
            return False

        chunk_index = 0
        while True:
            chunk_data, frames_in_chunk, ret = [], 0, True
            while frames_in_chunk < FRAME_CHUNK_SIZE:
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_number % frame_skip == 0:
                    elapsed_seconds = frame_number / fps
                    frame_timestamp = video_start_time + timedelta(seconds=elapsed_seconds)
                    o2_idx = df_arduino.index.get_indexer([frame_timestamp], method="nearest")[0]
                    t_idx = df_temp.index.get_indexer([frame_timestamp], method="nearest")[0]
                    if o2_idx != -1 and t_idx != -1:
                        resized = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        chunk_data.append([exp_id, frame_timestamp, elapsed_seconds,
                                           df_temp.iloc[t_idx]["temperature_C"],
                                           df_arduino.iloc[o2_idx]["oxygen_umol_L"]] + rgb.flatten().tolist())
                frames_in_chunk += 1
            if not chunk_data:
                break
            pd.DataFrame(chunk_data).to_parquet(os.path.join(temp_dir, f"{exp_id}_chunk_{chunk_index}.parquet"))
            chunk_index += 1
            if not ret:
                break
        cap.release()
    except Exception:
        traceback.print_exc()
        return False
    return True


def create_pinn_dataframe(experiments: list, output_path: str, frame_skip: int, resize_dim: tuple) -> pd.DataFrame:
    if os.path.exists(output_path):
        print(f"Loading cached feature dataframe: {output_path}")
        return pd.read_parquet(output_path)

    temp_dir = os.path.join(os.path.dirname(output_path), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    tasks = [{"exp": exp, "frame_skip": frame_skip, "resize_dim": resize_dim, "temp_dir": temp_dir}
             for exp in experiments]
    with Pool(processes=max(1, NUM_WORKERS)) as pool:
        list(tqdm(pool.imap_unordered(process_experiment_chunked, tasks), total=len(tasks),
                  desc="Processing experiments"))
    chunk_files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    if not chunk_files:
        raise ValueError("No data chunks were generated.")
    master_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    shutil.rmtree(temp_dir)

    pixel_cols = [f"pixel_{i}" for i in range(resize_dim[0] * resize_dim[1] * 3)]
    master_df.columns = ["experiment_id", "timestamp", "elapsed_seconds", "temperature_C", "oxygen_umol_L"] + pixel_cols
    for col in pixel_cols:
        master_df[col] = pd.to_numeric(master_df[col], downcast="unsigned")
    master_df["elapsed_seconds"] = master_df["elapsed_seconds"].astype("float32")
    master_df["temperature_C"] = master_df["temperature_C"].astype("float32")
    master_df.to_parquet(output_path, index=False)
    print(f"Saved feature dataframe with {len(master_df)} frames to {output_path}")
    return master_df


def save_plot_and_data(fig, plot_name: str, data_df: Optional[pd.DataFrame], output_dir: str):
    """Save every figure together with the data behind it."""
    os.makedirs(output_dir, exist_ok=True)
    if fig is not None:
        fig.savefig(os.path.join(output_dir, f"{plot_name}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    if data_df is not None and not data_df.empty:
        data_df.to_csv(os.path.join(output_dir, f"{plot_name}_data.csv"), index=False)


# ==============================================================================
# STAGE 2: DATASET AND MODEL
# ==============================================================================

class FrameStore:
    """Holds the frames once in memory; datasets below are views with their own transform."""

    def __init__(self, df: pd.DataFrame, exp_id_map: dict, storage_dim: tuple):
        pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
        self.pixels = df[pixel_cols].to_numpy(dtype=np.uint8).reshape(-1, storage_dim[0], storage_dim[1], 3)
        self.oxygen = df["oxygen_umol_L"].to_numpy(dtype=np.float32)
        self.temperature = df["temperature_C"].to_numpy(dtype=np.float32)
        self.exp_index = df["experiment_id"].map(exp_id_map).to_numpy(dtype=np.int64)


class VideoFrameDataset(Dataset):
    """Returns (image, DO, experiment index, temperature). Each instance has its own transform."""

    def __init__(self, store: FrameStore, model_input_dim: tuple, transform=None):
        self.store = store
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(model_input_dim, antialias=True)
        self.transform = transform

    def __len__(self):
        return len(self.store.oxygen)

    def __getitem__(self, idx: int):
        image = self.resize(self.to_tensor(self.store.pixels[idx]))
        if self.transform is not None:
            image = self.transform(image)
        return (image,
                torch.tensor(self.store.oxygen[idx], dtype=torch.float32),
                torch.tensor(self.store.exp_index[idx], dtype=torch.long),
                torch.tensor(self.store.temperature[idx], dtype=torch.float32))


class AlgaePINN(nn.Module):
    """ViT backbone; CLS token + temperature -> DO; patch tokens -> mask and confidence map;
    experiment embedding -> two-site Stern-Volmer parameters (used only in the physics loss)."""

    def __init__(self, num_experiments: int, embedding_dim: int = 8):
        super().__init__()
        self.vit_base = timm.create_model(TRANSFER_MODEL_NAME, pretrained=USE_TRANSFER_LEARNING, num_classes=0)
        self.vit_embed_dim = self.vit_base.embed_dim
        ps = self.vit_base.patch_embed.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps
        self.grid_size = MODEL_INPUT_DIM[0] // patch_size

        self.experiment_embedding = nn.Embedding(num_experiments, embedding_dim)
        self.sv_param_head = nn.Sequential(nn.Linear(embedding_dim, 16), nn.GELU(), nn.Linear(16, 4))
        self.oxygen_head = nn.Sequential(nn.Linear(self.vit_embed_dim + 1, 128), nn.GELU(), nn.Linear(128, 1))

        decoder_channels = 256
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.vit_embed_dim, decoder_channels, 1), nn.GELU(),
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_channels, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 1, 1), nn.Sigmoid())
        self.confidence_head = None
        if USE_CONFIDENCE_LOSS:
            self.confidence_head = nn.Sequential(
                nn.Conv2d(self.vit_embed_dim, decoder_channels, 1), nn.GELU(),
                nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
                nn.Conv2d(decoder_channels, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, image_tensor, exp_id_index, temperature):
        features = self.vit_base.forward_features(image_tensor)
        cls_token = features[:, 0]
        oxygen_pred = self.oxygen_head(torch.cat([cls_token, temperature.view(-1, 1)], dim=1))

        patch_tokens = features[:, 1:]
        B, N, C = patch_tokens.shape
        grid = patch_tokens.reshape(B, self.grid_size, self.grid_size, C).permute(0, 3, 1, 2).contiguous()
        mask_pred = self.mask_head(grid)

        sv_raw = self.sv_param_head(self.experiment_embedding(exp_id_index))
        sv_params = {"i0": nn.functional.softplus(sv_raw[:, 0]),
                     "ksv1": nn.functional.softplus(sv_raw[:, 1]),
                     "ksv2": nn.functional.softplus(sv_raw[:, 2]),
                     "a": torch.sigmoid(sv_raw[:, 3])}

        confidence_map = None
        if self.confidence_head is not None:
            confidence_map = self.confidence_head(grid).squeeze(1)
            sums = confidence_map.view(B, -1).sum(dim=1).view(B, 1, 1)
            confidence_map = confidence_map / (sums + 1e-8)  # sums to 1 over the image

        return {"oxygen_pred": oxygen_pred.squeeze(-1), "mask_pred": mask_pred.squeeze(1),
                "sv_params": sv_params, "confidence_map": confidence_map}


# ==============================================================================
# STAGE 3: LOSS, TRAINING, EVALUATION
# ==============================================================================

def calculate_pinn_loss(model_outputs: dict, batch_data: list, lambda_physics: float) -> dict:
    """L_total = MSE(DO_pred, DO_true) + lambda * mean(C * (M * R)^2),
    R = |I_obs - I_SV(DO_pred)| / (I_obs + 1)  (two-site Stern-Volmer, relative form).
    I_obs is channel 0 (red) of the network input tensor, scaled by 255."""
    image_tensor, oxygen_true, _, _ = batch_data
    oxygen_pred = model_outputs["oxygen_pred"]
    mask, sv, conf = model_outputs["mask_pred"], model_outputs["sv_params"], model_outputs["confidence_map"]

    loss_data = nn.functional.mse_loss(oxygen_pred, oxygen_true)

    red = transforms.functional.resize(image_tensor, mask.shape[-2:], antialias=True)[:, 0] * 255.0
    do_r, i0 = oxygen_pred.view(-1, 1, 1), sv["i0"].view(-1, 1, 1)
    if PHYSICS_MODEL == "nonlinear":
        a, k1, k2 = sv["a"].view(-1, 1, 1), sv["ksv1"].view(-1, 1, 1), sv["ksv2"].view(-1, 1, 1)
        predicted_intensity = i0 * (a / (1 + k1 * do_r) + (1 - a) / (1 + k2 * do_r))
        residual = torch.abs(red - predicted_intensity) / (red + 1.0)
    else:
        residual = torch.abs(i0 / (red + 1e-6) - (1 + sv["ksv1"].view(-1, 1, 1) * do_r))

    squared = (mask * residual) ** 2
    loss_physics = torch.mean(conf * squared) if conf is not None else torch.mean(squared)
    total = loss_data + lambda_physics * loss_physics
    return {"total": total, "data": loss_data.detach(), "physics": loss_physics.detach()}


def train_one_epoch(model, loader, optimizer, lambda_physics, scaler) -> dict:
    model.train()
    sums = {"total": 0.0, "data": 0.0, "physics": 0.0}
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        batch_data = [d.to(DEVICE, non_blocking=True) for d in batch]
        with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
            outputs = model(batch_data[0], batch_data[2], batch_data[3])
            loss_dict = calculate_pinn_loss(outputs, batch_data, lambda_physics)
            loss = loss_dict["total"] / GRAD_ACCUMULATION_STEPS
        if torch.isnan(loss):
            warnings.warn(f"NaN loss at batch {i}; skipped.")
            continue
        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            if USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for k in sums:
            if not torch.isnan(loss_dict[k]):
                sums[k] += loss_dict[k].item()
    return {k: v / len(loader) for k, v in sums.items()}


def predict(model, loader) -> dict:
    model.eval()
    out = {"o2_true": [], "o2_pred": [], "biofouling_score": [], "temperature_C": []}
    with torch.no_grad():
        for batch in loader:
            image, o2_true, exp_idx, temp = [d.to(DEVICE, non_blocking=True) for d in batch]
            with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
                outputs = model(image, exp_idx, temp)
            out["o2_true"].append(o2_true.cpu().numpy())
            out["o2_pred"].append(outputs["oxygen_pred"].float().cpu().numpy())
            out["biofouling_score"].append((1.0 - outputs["mask_pred"].float().mean(dim=[1, 2])).cpu().numpy())
            out["temperature_C"].append(temp.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate_mae(model, loader) -> float:
    res = predict(model, loader)
    if np.isnan(res["o2_pred"]).any():
        return np.nan
    return float(mean_absolute_error(res["o2_true"], res["o2_pred"]))


def make_loader(dataset, indices, shuffle: bool) -> DataLoader:
    return DataLoader(Subset(dataset, indices), BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      worker_init_fn=seed_worker, pin_memory=True)


# ==============================================================================
# STAGE 4: HYPERPARAMETER OPTIMISATION (once, before the LOOCV)
# ==============================================================================

def run_optuna_tuning(store: FrameStore, exp_labels: np.ndarray, num_experiments: int,
                      hpo_dir: str) -> dict:
    """Run Optuna once (or load its cached result from hpo_dir)."""
    params_path = os.path.join(hpo_dir, "best_hpo_params.json")
    db_path = os.path.join(hpo_dir, "hpo_study.db")
    os.makedirs(hpo_dir, exist_ok=True)
    if os.path.exists(params_path):
        print(f"Loading cached HPO parameters: {params_path}")
        with open(params_path) as f:
            return json.load(f)

    # HPO frames are passed without ImageNet normalisation.
    hpo_dataset = VideoFrameDataset(store, MODEL_INPUT_DIM, transform=None)
    subset_idx, _ = train_test_split(np.arange(len(hpo_dataset)), train_size=HPO_DATA_SUBSET_FRACTION,
                                     stratify=exp_labels, random_state=SEED)
    tr_pos, va_pos = train_test_split(np.arange(len(subset_idx)), test_size=0.2, random_state=SEED)
    tr_idx, va_idx = subset_idx[tr_pos], subset_idx[va_pos]

    def objective(trial):
        try:
            p = {"lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                 "lambda_physics": trial.suggest_float("lambda_physics", 0.1, 10.0, log=True),
                 "embedding_dim": trial.suggest_categorical("embedding_dim", [8, 16]),
                 "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)}
            model = AlgaePINN(num_experiments, p["embedding_dim"]).to(DEVICE)
            opt = optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
            scaler = torch.amp.GradScaler(enabled=USE_AMP)
            tl, vl = make_loader(hpo_dataset, tr_idx, True), make_loader(hpo_dataset, va_idx, False)
            val_mae = np.nan
            for epoch in range(N_EPOCHS_FOR_TUNING):
                train_one_epoch(model, tl, opt, p["lambda_physics"], scaler)
                val_mae = evaluate_mae(model, vl)
                if np.isnan(val_mae):
                    return 1e9
                trial.report(val_mae, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return val_mae
        except optuna.exceptions.TrialPruned:
            raise
        except Exception:
            traceback.print_exc()
            return 1e9
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(),
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=f"sqlite:///{db_path}", study_name="pvit_hpo", load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if N_TRIALS_FOR_TUNING - done > 0:
        study.optimize(objective, n_trials=N_TRIALS_FOR_TUNING - done, show_progress_bar=True)
    if study.best_value >= 1e9:
        raise RuntimeError("HPO failed: no successful trial.")
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    study.trials_dataframe().to_csv(os.path.join(hpo_dir, "hpo_trials_history.csv"), index=False)
    return study.best_params


# ==============================================================================
# STAGE 5: FOLD PLAN, MEMBER TRAINING, SELECTION, TEST
# ==============================================================================

def build_fold_plan(exp_ids: list, exp_dates: dict) -> pd.DataFrame:
    """One fold per day, in chronological order. Validation day = preceding day
    (the following day for the first day)."""
    chrono = sorted(exp_ids, key=lambda e: exp_dates[e])
    rows = []
    for pos, test_day in enumerate(chrono):
        val_day = chrono[pos - 1] if pos > 0 else chrono[1]
        rows.append({"fold": pos + 1, "test_day": test_day, "val_day": val_day})
    plan = pd.DataFrame(rows)

    for _, r in plan.iterrows():
        assert r.test_day in exp_ids and r.val_day in exp_ids, f"Unknown day in fold plan: {r.to_dict()}"
        assert r.test_day != r.val_day, f"Fold {r.fold}: validation day equals test day"
    plan["test_date"] = plan["test_day"].map(lambda e: exp_dates[e].strftime("%Y-%m-%d"))
    plan["val_date"] = plan["val_day"].map(lambda e: exp_dates[e].strftime("%Y-%m-%d"))
    plan["n_train_days"] = len(exp_ids) - 2
    return plan


def lambda_for_epoch(base_lambda: float, epoch: int) -> float:
    if not USE_CURRICULUM_LEARNING:
        return base_lambda
    ramp = min(1.0, epoch / LAMBDA_CURRICULUM_EPOCHS)
    return base_lambda * (LAMBDA_CURRICULUM_START + (1 - LAMBDA_CURRICULUM_START) * ramp)


def train_member(num_experiments, best_params, train_loader, val_loader, weights_path, fold_id, member,
                 analysis_dir) -> dict:
    """Train one ensemble member on the training days and save the weights of the epoch with
    the lowest validation-day MAE. The test day is not available inside this function."""
    model = AlgaePINN(num_experiments, best_params["embedding_dim"]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"],
                            weight_decay=best_params.get("weight_decay", 1e-4))
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_FOR_FINAL_TRAINING)
                 if USE_LR_SCHEDULER else None)

    best_val, best_epoch, kept_state, history = float("inf"), None, None, []
    for epoch in range(1, N_EPOCHS_FOR_FINAL_TRAINING + 1):
        losses = train_one_epoch(model, train_loader, optimizer,
                                 lambda_for_epoch(best_params["lambda_physics"], epoch), scaler)
        val_mae = evaluate_mae(model, val_loader)
        history.append({"fold": fold_id, "member": member, "epoch": epoch,
                        **{f"train_loss_{k}": v for k, v in losses.items()}, "val_mae": val_mae})
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            # Independent copy of this epoch's weights (state_dict() alone would keep
            # pointing at the live parameters and follow later updates).
            kept_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scheduler:
            scheduler.step()
        print(f"    Ep {epoch:02d}: loss={losses['total']:.4f}  val MAE={val_mae:.4f}", end="\r")
    print()

    if kept_state is None:
        raise RuntimeError(f"Fold {fold_id}, member {member}: validation MAE was NaN at every epoch.")
    torch.save(kept_state, weights_path)

    hist = pd.DataFrame(history)
    hist["kept_epoch"] = best_epoch
    hist.to_csv(weights_path.replace(".pth", "_history.csv"), index=False)
    plot_member_history(hist, fold_id, member, analysis_dir)
    return {"kept_epoch": int(best_epoch)}


def run_fold(row, store, train_ds, eval_ds, filtered_df, num_experiments, best_params, exp_id_map,
             cache_dir, analysis_dir, red_heatmap_dir) -> tuple:
    fold_id, test_day, val_day = int(row.fold), row.test_day, row.val_day
    print(f"\n=== Fold {fold_id} | test: {test_day} | validation: {val_day} ===")
    fold_dir = os.path.join(cache_dir, f"fold_{fold_id:02d}")
    os.makedirs(fold_dir, exist_ok=True)
    pred_path = os.path.join(fold_dir, "test_predictions.parquet")
    summ_path = os.path.join(fold_dir, "fold_summary.json")
    if os.path.exists(pred_path) and os.path.exists(summ_path) and not FORCE_RERUN_CV:
        print("  Cached fold results found; loading.")
        with open(summ_path) as f:
            return pd.read_parquet(pred_path), json.load(f)

    exp_col = filtered_df["experiment_id"]
    test_idx = np.flatnonzero((exp_col == test_day).to_numpy())
    val_idx = np.flatnonzero((exp_col == val_day).to_numpy())
    train_idx = np.flatnonzero(((exp_col != test_day) & (exp_col != val_day)).to_numpy())
    assert not (set(train_idx) & set(val_idx)) and not (set(train_idx) & set(test_idx)) \
        and not (set(val_idx) & set(test_idx)), "Train/validation/test overlap"

    train_loader = make_loader(train_ds, train_idx, shuffle=True)
    val_loader = make_loader(eval_ds, val_idx, shuffle=False)

    # --- Steps 1-2: train members, keep best-validation epoch (training + validation days only) ---
    weight_paths, kept_epochs = [], []
    for member in range(ENSEMBLE_SIZE):
        path = os.path.join(fold_dir, f"member_{member}.pth")
        weight_paths.append(path)
        hist_path = path.replace(".pth", "_history.csv")
        if os.path.exists(path) and not FORCE_RERUN_CV:
            print(f"  Member {member + 1}/{ENSEMBLE_SIZE}: cached weights found.")
            kept_epochs.append(int(pd.read_csv(hist_path).kept_epoch.iloc[0]) if os.path.exists(hist_path) else None)
            continue
        print(f"  Training member {member + 1}/{ENSEMBLE_SIZE} ...")
        info = train_member(num_experiments, best_params, train_loader, val_loader, path, fold_id, member,
                            analysis_dir)
        kept_epochs.append(info["kept_epoch"])

    # --- Validation-day MAE of each member's checkpoint (used to choose the selected member) ---
    members, val_maes = [], []
    for path in weight_paths:
        m = AlgaePINN(num_experiments, best_params["embedding_dim"]).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        members.append(m)
        val_maes.append(evaluate_mae(m, val_loader))
    if all(np.isnan(v) for v in val_maes):
        raise RuntimeError(f"Fold {fold_id}: all members produced NaN on the validation day.")
    selected = int(np.nanargmin(val_maes))
    print(f"  Validation MAE per member: {np.round(val_maes, 4).tolist()} -> selected member {selected}")

    # --- Step 3: test day is loaded and evaluated only now, once ---
    test_loader = make_loader(eval_ds, test_idx, shuffle=False)
    member_preds = [predict(m, test_loader) for m in members]
    sel = member_preds[selected]
    res = pd.DataFrame({"fold": fold_id, "original_index": test_idx,
                        "o2_true": sel["o2_true"], "o2_pred": sel["o2_pred"],
                        "biofouling_score": sel["biofouling_score"], "temperature_C": sel["temperature_C"]})
    stack = np.stack([p["o2_pred"] for p in member_preds])
    for k in range(ENSEMBLE_SIZE):
        res[f"o2_pred_member_{k}"] = stack[k]
    res["o2_pred_ensemble_mean"] = stack.mean(axis=0)
    res["o2_pred_ensemble_std"] = stack.std(axis=0)
    res.to_parquet(pred_path, index=False)

    summary = {"fold": fold_id, "test_day": test_day, "val_day": val_day,
               "n_train_frames": int(len(train_idx)), "n_val_frames": int(len(val_idx)),
               "n_test_frames": int(len(test_idx)),
               "kept_epoch_per_member": kept_epochs,
               "val_mae_per_member": [float(v) for v in val_maes], "selected_member": selected,
               "test_mae_per_member": [float(mean_absolute_error(res.o2_true, stack[k]))
                                       for k in range(ENSEMBLE_SIZE)],
               "test_mae_ensemble_mean": float(mean_absolute_error(res.o2_true, res.o2_pred_ensemble_mean)),
               "test_mae_selected_member": float(mean_absolute_error(res.o2_true, res.o2_pred)),
               "checkpoint": "best validation-day MAE epoch"}
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Test MAE  ensemble mean: {summary['test_mae_ensemble_mean']:.4f} | "
          f"validation-selected member: {summary['test_mae_selected_member']:.4f}")

    plot_confidence_map(members[selected], test_loader, f"fold_{fold_id:02d}", analysis_dir)
    plot_biofouling_correlation(members[selected], test_loader, exp_id_map, red_heatmap_dir,
                                f"fold_{fold_id:02d}", analysis_dir)
    del members
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res, summary


# ==============================================================================
# STAGE 6: REPORTING
# ==============================================================================

def plot_member_history(hist: pd.DataFrame, fold_id: int, member: int, analysis_dir: str):
    out = os.path.join(analysis_dir, "1_training_history")
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(hist.epoch, hist.val_mae, color="tab:orange", marker="o", label="Validation-day MAE")
    ax1.axvline(hist.kept_epoch.iloc[0], color="grey", linestyle=":", label="Best-validation epoch (kept)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation MAE (µmol/L)", color="tab:orange")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(hist.epoch, hist.train_loss_physics, color="tab:green", linestyle="--", label="Train physics loss")
    ax2.set_yscale("log")
    ax2.set_ylabel("Train physics loss", color="tab:green")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Fold {fold_id}, member {member}")
    save_plot_and_data(fig, f"fold_{fold_id:02d}_member_{member}_history", hist, out)


def plot_confidence_map(model, loader, prefix: str, analysis_dir: str):
    if not USE_CONFIDENCE_LOSS:
        return
    model.eval()
    with torch.no_grad():
        image, _, exp_idx, temp = [d.to(DEVICE) for d in next(iter(loader))]
        conf = model(image, exp_idx, temp)["confidence_map"].mean(dim=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(conf, cmap="viridis", ax=ax, cbar_kws={"label": "Confidence weight (normalised)"})
    ax.set_title(f"{prefix}: mean physics confidence map (test day)")
    save_plot_and_data(fig, f"{prefix}_confidence_map", pd.DataFrame(conf),
                       os.path.join(analysis_dir, "2_confidence_maps"))


def plot_biofouling_correlation(model, loader, exp_id_map, red_heatmap_dir, prefix, analysis_dir):
    out = os.path.join(analysis_dir, "3_biofouling_analysis")
    model.eval()
    masks, idxs = [], []
    with torch.no_grad():
        for batch in loader:
            image, _, exp_idx, temp = [d.to(DEVICE) for d in batch]
            masks.append(model(image, exp_idx, temp)["mask_pred"].cpu().numpy())
            idxs.append(exp_idx.cpu().numpy())
    masks, idxs = np.concatenate(masks), np.concatenate(idxs)
    idx_to_id = {v: k for k, v in exp_id_map.items()}
    for e_idx in np.unique(idxs):
        exp_id = idx_to_id[int(e_idx)]
        biofouling = 1.0 - masks[idxs == e_idx].mean(axis=0)
        csv = os.path.join(red_heatmap_dir, f"heatmap_{exp_id}_overall_data.csv")
        if not os.path.exists(csv):
            continue
        red = pd.read_csv(csv, header=None).values
        if red.shape != biofouling.shape:
            red = cv2.resize(red, (biofouling.shape[1], biofouling.shape[0]), interpolation=cv2.INTER_LINEAR)
        p, r = biofouling.flatten(), red.flatten()
        corr, pval = scipy.stats.pearsonr(p, r) if p.std() > 1e-6 and r.std() > 1e-6 else (0.0, 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle(f"{exp_id} ({prefix}) - Pearson r = {corr:.3f} (p = {pval:.2e})")
        sns.heatmap(biofouling, cmap="YlGnBu", ax=axes[0]); axes[0].set_title("Predicted biofouling (1 - mask)")
        sns.heatmap(red, cmap="Reds_r", ax=axes[1]); axes[1].set_title("Mean red intensity")
        sns.regplot(x=r, y=p, ax=axes[2], scatter_kws={"alpha": 0.1, "s": 10})
        axes[2].set_xlabel("Red intensity"); axes[2].set_ylabel("Predicted biofouling")
        save_plot_and_data(fig, f"{prefix}_{exp_id}_biofouling_correlation",
                           pd.DataFrame({"predicted_biofouling": p, "red_intensity": r}), out)


def generate_red_intensity_heatmaps(master_df, experiments, resize_dim, report_dir):
    if os.path.exists(report_dir):
        return
    os.makedirs(report_dir, exist_ok=True)
    pixel_cols = [c for c in master_df.columns if c.startswith("pixel_")]
    for exp in tqdm(experiments, desc="Red-intensity heatmaps"):
        exp_df = master_df[master_df.experiment_id == exp["id"]]
        maps = []
        for s, e in ANALYSIS_INTERVALS_S:
            iv = exp_df[(exp_df.elapsed_seconds >= s) & (exp_df.elapsed_seconds <= e)]
            if iv.empty:
                continue
            imgs = iv[pixel_cols].values.astype(np.uint8).reshape(-1, resize_dim[0], resize_dim[1], 3)
            maps.append(imgs[..., 0].mean(axis=0))
        if maps:
            overall = np.mean(maps, axis=0)
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(overall, cmap="Reds_r", ax=ax)
            ax.set_title(f"Mean red intensity: {exp['id']}")
            save_plot_and_data(fig, f"heatmap_{exp['id']}_overall", pd.DataFrame(overall), report_dir)


def final_report(oof: pd.DataFrame, summaries: list, output_dir: str, analysis_dir: str):
    """Report both test results: (a) ensemble mean and (b) validation-selected member."""
    out = os.path.join(analysis_dir, "4_final_results")
    os.makedirs(out, exist_ok=True)
    summ = pd.DataFrame(summaries).sort_values("fold").reset_index(drop=True)
    summ.to_csv(os.path.join(out, "loocv_fold_summary.csv"), index=False)
    oof.to_parquet(os.path.join(output_dir, "loocv_test_predictions.parquet"), index=False)

    variants = {"ensemble_mean": ("o2_pred_ensemble_mean", "test_mae_ensemble_mean", "Ensemble mean"),
                "validation_selected_member": ("o2_pred", "test_mae_selected_member",
                                               "Validation-selected member")}
    metrics = {"protocol": f"LOOCV over {len(summ)} days; 1 validation day; {ENSEMBLE_SIZE} members",
               "validation_day": "chronologically preceding day (following day for the first day)",
               "checkpoint": "best validation-day MAE epoch per member", "n_folds": int(len(summ)),
               "n_test_frames": int(len(oof))}
    for key, (pred_col, fold_col, label) in variants.items():
        err = oof[pred_col] - oof.o2_true
        metrics[key] = {
            "mean_of_per_day_test_mae": float(summ[fold_col].mean()),
            "std_of_per_day_test_mae": float(summ[fold_col].std(ddof=1)),
            "pooled_test_mae": float(err.abs().mean()),
            "pooled_error_mean": float(err.mean()),
            "pooled_error_std": float(err.std(ddof=1)),
        }
        m = metrics[key]

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(x=oof.o2_true, y=oof[pred_col], alpha=0.5, ax=ax)
        lo, hi = oof.o2_true.min(), oof.o2_true.max()
        ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
        ax.set_title(f"LOOCV test parity: {label}\n"
                     f"mean of per-day MAE = {m['mean_of_per_day_test_mae']:.2f} µmol/L; "
                     f"pooled MAE = {m['pooled_test_mae']:.2f} µmol/L")
        ax.set_xlabel("True DO (µmol/L)"); ax.set_ylabel("Predicted DO (µmol/L)"); ax.legend()
        save_plot_and_data(fig, f"loocv_parity_{key}",
                           oof[["fold", "experiment_id", "o2_true", pred_col]].rename(columns={pred_col: "o2_pred"}),
                           out)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(err, kde=True, bins=50, ax=ax)
        ax.set_title(f"Test error distribution ({label}): mean {m['pooled_error_mean']:.2f}, "
                     f"std {m['pooled_error_std']:.2f} µmol/L")
        ax.set_xlabel("Prediction error (µmol/L)")
        save_plot_and_data(fig, f"loocv_error_histogram_{key}", pd.DataFrame({"error": err}), out)

    with open(os.path.join(out, "loocv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    by_day = summ.melt(id_vars=["fold", "test_day"],
                       value_vars=["test_mae_ensemble_mean", "test_mae_selected_member"],
                       var_name="result", value_name="test_mae")
    by_day["result"] = by_day["result"].map({"test_mae_ensemble_mean": "Ensemble mean",
                                             "test_mae_selected_member": "Validation-selected member"})
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=by_day, x="test_day", y="test_mae", hue="result", ax=ax)
    ax.set_ylabel("Test MAE (µmol/L)"); ax.set_xlabel("Held-out test day")
    ax.tick_params(axis="x", rotation=45)
    save_plot_and_data(fig, "loocv_test_mae_by_day", by_day, out)

    print("\n--- LOOCV results ---")
    print(f"  folds: {len(summ)} | validation day: preceding day | checkpoint: best-validation epoch")
    for key, (_, _, label) in variants.items():
        m = metrics[key]
        print(f"  {label:28s} mean of per-day MAE = {m['mean_of_per_day_test_mae']:.3f} "
              f"(SD {m['std_of_per_day_test_mae']:.3f}) | pooled MAE = {m['pooled_test_mae']:.3f} µmol/L")


# ==============================================================================
# MAIN
# ==============================================================================

def load_cached_fold(fold_id: int, cache_dir: str):
    fold_dir = os.path.join(cache_dir, f"fold_{fold_id:02d}")
    pred_path, summ_path = (os.path.join(fold_dir, "test_predictions.parquet"),
                            os.path.join(fold_dir, "fold_summary.json"))
    if not (os.path.exists(pred_path) and os.path.exists(summ_path)):
        return None
    with open(summ_path) as f:
        return pd.read_parquet(pred_path), json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Folder containing one sub-folder per day.")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder for caches and results.")
    ap.add_argument("--hyperparameters", default=None,
                    help="Optional JSON with lr, lambda_physics, embedding_dim, weight_decay; skips Optuna.")
    ap.add_argument("--folds", default=None,
                    help="Optional, e.g. --folds 1,2,3 to run only those folds in this job. "
                         "Without it all 13 folds run one after the other.")
    args = ap.parse_args(argv)

    start = datetime.now()
    output_dir = os.path.abspath(args.output_dir)
    analysis_dir = os.path.join(output_dir, "analysis")
    cache_dir = os.path.join(output_dir, "fold_cache")
    feature_path = os.path.join(output_dir, "features", "pinn_features.parquet")
    hpo_dir = os.path.join(output_dir, "hpo")
    red_heatmap_dir = os.path.join(output_dir, "red_intensity_heatmaps")
    for d in (output_dir, analysis_dir, cache_dir, os.path.dirname(feature_path)):
        os.makedirs(d, exist_ok=True)

    print(f"Device: {DEVICE}\nOutput folder: {output_dir}")
    experiments = find_experiment_files(args.data_root)
    master_df = create_pinn_dataframe(experiments, feature_path, FRAME_SKIP, STORAGE_RESIZE_DIM)
    generate_red_intensity_heatmaps(master_df, experiments, STORAGE_RESIZE_DIM, red_heatmap_dir)

    in_window = functools.reduce(np.logical_or, [(master_df.elapsed_seconds >= s) & (master_df.elapsed_seconds <= e)
                                                 for s, e in ANALYSIS_INTERVALS_S])
    filtered_df = master_df[in_window].reset_index(drop=True)
    del master_df
    if filtered_df.empty:
        raise RuntimeError("No frames inside the analysis intervals.")

    exp_ids = [str(e) for e in filtered_df.experiment_id.unique()]
    if len(exp_ids) < 3:
        raise ValueError(f"Need at least 3 experimental days, found {len(exp_ids)}.")
    exp_id_map = {e: i for i, e in enumerate(exp_ids)}
    # Day order comes from the frame table itself (first time stamp of each day), so it does not
    # depend on the current folder names matching the names stored in the cached table.
    first_ts = pd.to_datetime(filtered_df["timestamp"]).groupby(filtered_df["experiment_id"].astype(str)).min()
    exp_dates = {e: first_ts[e] for e in exp_ids}
    folder_ids = {e["id"] for e in experiments}
    not_in_folders = [e for e in exp_ids if e not in folder_ids]
    if not_in_folders:
        print(f"Note: days in the frame table without a matching folder name: {not_in_folders} "
              "(used as they are in the table).")

    plan = build_fold_plan(exp_ids, exp_dates)
    plan.to_csv(os.path.join(output_dir, "fold_plan.csv"), index=False)
    print(plan[["fold", "test_day", "val_day"]].to_string(index=False))

    store = FrameStore(filtered_df, exp_id_map, STORAGE_RESIZE_DIM)
    if args.hyperparameters:
        with open(args.hyperparameters) as f:
            best_params = json.load(f)
        print(f"Using hyperparameters from {args.hyperparameters}")
    else:
        best_params = run_optuna_tuning(store, filtered_df.experiment_id.values, len(exp_ids), hpo_dir)
    print(f"Hyperparameters: {best_params}")

    train_tf = (transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                    IMAGENET_NORMALIZE]) if USE_DATA_AUGMENTATION else IMAGENET_NORMALIZE)
    train_ds = VideoFrameDataset(store, MODEL_INPUT_DIM, transform=train_tf)
    eval_ds = VideoFrameDataset(store, MODEL_INPUT_DIM, transform=IMAGENET_NORMALIZE)

    run_plan = plan
    if args.folds:
        wanted = {int(f) for f in args.folds.split(",") if f.strip()}
        run_plan = plan[plan.fold.isin(wanted)]
        if run_plan.empty:
            raise ValueError(f"--folds {args.folds} matches no fold (folds are 1-{len(plan)}).")

    for row in run_plan.itertuples(index=False):
        run_fold(row, store, train_ds, eval_ds, filtered_df, len(exp_ids), best_params,
                 exp_id_map, cache_dir, analysis_dir, red_heatmap_dir)

    # Final report as soon as every fold has a finished result (from this job or earlier ones).
    cached = {int(f): load_cached_fold(int(f), cache_dir) for f in plan.fold}
    missing = [f for f, c in cached.items() if c is None]
    if missing:
        print(f"\nFolds still to run: {missing}. The final report is written when all folds are done.")
        return
    oof = pd.concat([c[0] for c in cached.values()]).sort_values("original_index").reset_index(drop=True)
    meta = filtered_df[["timestamp", "experiment_id", "elapsed_seconds"]].iloc[oof.original_index].reset_index(drop=True)
    oof = pd.concat([meta, oof], axis=1)
    final_report(oof, [c[1] for c in cached.values()], output_dir, analysis_dir)
    print(f"\nFinished in {datetime.now() - start}. Results in {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
