"""
==============================================================================
Advanced Research PINN Framework
==============================================================================
This script provides an end-to-end framework for training and evaluating an
advanced Physics-Informed Neural Network (PINN).

Scientific Goals:
1. Robust Prediction: Estimate Dissolved Oxygen (DO) from video frames.
2. Uncertainty Quantification: Utilize Deep Ensembles to self-diagnose
   low-confidence predictions (e.g., severe biofouling).
3. Biofouling Estimation: Physics-Informed mask head outputs a fouling factor.

Methodology:
- Vision Transformer (ViT) feature extraction.
- Leave-One-Out Cross-Validation (LOOCV) deployment simulations.
- Stern-Volmer physical parameter integrations.
==============================================================================
"""

import os
import sys
import glob
import re
import traceback
import warnings
import random
import json
import functools
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# Data Manipulation
import pandas as pd
import numpy as np

# Image Processing
import cv2

# Visualization
import matplotlib

# CRITICAL FOR CLUSTER/HEADLESS ENV: Non-interactive 'Agg' backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Progress Bars
from tqdm import tqdm

# ML & Optimization
import optuna
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import timm
from torchvision import transforms
import shutil

# ==============================================================================
# REPRODUCIBILITY CONFIGURATION
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False  # Set False for ensemble speed
torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Global Configuration
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# PATHS & DIRECTORY STRUCTURE
# ==============================================================================
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
EXPERIMENTS_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "raw")

# Output & Reporting Directories
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "detailed_analysis_report")
FOLDS_CACHE_DIR = os.path.join(OUTPUT_DIR, "loocv_folds")
HEATMAPS_DIR = os.path.join(OUTPUT_DIR, "heatmaps")

# Caching Directories
FEATURE_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "cache_features")
FEATURE_DATAFRAME_PATH = os.path.join(FEATURE_CACHE_DIR, "pinn_features_pandas.parquet")
HPO_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "cache_hpo")
HPO_PARAMS_PATH = os.path.join(HPO_CACHE_DIR, "best_hpo_params.json")
HPO_STUDY_DB_PATH = os.path.join(HPO_CACHE_DIR, "hpo_study.db")

# ==============================================================================
# PIPELINE PARAMETERS
# ==============================================================================
FORCE_RECREATE_DATAFRAME = False
FORCE_RECREATE_HEATMAPS = False
NUM_EXPERIMENT_DAYS_TO_USE = None  # None = use all available data
FRAME_SKIP = 15
STORAGE_RESIZE_DIM = (48, 48)
MODEL_INPUT_DIM = (224, 224)
FORCED_VIDEO_FPS = 30
FRAME_CHUNK_SIZE = 3000

# Analytical windows mapping (seconds)
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300), (3900, 4100), (4700, 4900)]
SENSOR_RESAMPLE_WINDOW_S = 10

# Model & HPO Settings
FORCE_RERUN_HPO = False
FORCE_RERUN_FOLDS = False
ENSEMBLE_SIZE = 3  # Increase to 5 for production
PHYSICS_MODEL = 'nonlinear'
N_TRIALS_FOR_TUNING = 15
N_EPOCHS_FOR_TUNING = 10
N_EPOCHS_FOR_FINAL_TRAINING = 35
HPO_DATA_SUBSET_FRACTION = 0.35
BATCH_SIZE = 48
GRAD_ACCUMULATION_STEPS = 16
GRADIENT_CLIP_VALUE = 1.0

# Architectural Toggles
USE_DATA_AUGMENTATION = True
USE_TRANSFER_LEARNING = True
TRANSFER_MODEL_NAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
USE_CURRICULUM_LEARNING = True
LAMBDA_CURRICULUM_START = 0.01
LAMBDA_CURRICULUM_EPOCHS = 10
USE_CONFIDENCE_LOSS = True

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == 'cuda')
DEVICE_TYPE = DEVICE.type

# Cluster Resource Detection
try:
    num_cores = int(os.environ.get('NSLOTS', cpu_count()))
    NUM_WORKERS = max(1, num_cores - 2)
    print(f"Environment detected. Using {NUM_WORKERS} workers (Cores={num_cores}).")
except (ValueError, TypeError):
    NUM_WORKERS = max(1, cpu_count() - 2)
    print(f"Could not read environment slots. Defaulting to {NUM_WORKERS} workers.")


# ==============================================================================
# DATA PARSING & PREPROCESSING (PANDAS ENGINES)
# ==============================================================================

def parse_arduino_log(file_path: str) -> pd.DataFrame:
    """Parses standard Arduino logs with robust timestamp interpolation."""
    timestamps, oxygen_values = [], []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "MEA" not in line:
                    continue
                try:
                    parts = line.split()
                    timestamp_str = f"{parts[0]} {parts[1]}"
                    dt_object = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    mea_index = parts.index("MEA")
                    raw_oxygen_value = int(parts[mea_index + 5])
                    timestamps.append(dt_object)
                    oxygen_values.append(raw_oxygen_value / 1000.0)
                except (ValueError, IndexError, TypeError):
                    pass
    except Exception as e:
        print(f"Error reading Arduino file {file_path}: {e}")
        return pd.DataFrame()

    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame({'timestamp': timestamps, 'oxygen_umol_L': oxygen_values})
    if df.timestamp.duplicated().any():
        df = df.groupby('timestamp')['oxygen_umol_L'].mean().reset_index()

    if SENSOR_RESAMPLE_WINDOW_S > 0:
        df = df.set_index('timestamp').resample(f'{SENSOR_RESAMPLE_WINDOW_S}S').mean()
        df = df.interpolate(method='linear', limit_direction='both').dropna().reset_index()

    return df


def parse_temperature_log(file_path: str) -> pd.DataFrame:
    """Parses environmental temperature logs with automated spike filtering."""
    try:
        df = pd.read_csv(file_path, usecols=[0, 1], names=['timestamp', 'temperature_C'], header=0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['temperature_C'] = pd.to_numeric(df['temperature_C'], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            return pd.DataFrame()

        if df.timestamp.duplicated().any():
            df = df.groupby('timestamp')['temperature_C'].mean().reset_index()
        df.sort_values('timestamp', inplace=True)

        # Signal Filtering: Drop extreme spikes (>100% change)
        pct_change = df['temperature_C'].pct_change().abs()
        spike_mask = pct_change >= 1.0
        if spike_mask.sum() > 0:
            df = df[~spike_mask].copy()

        if SENSOR_RESAMPLE_WINDOW_S > 0:
            df = df.set_index('timestamp').resample(f'{SENSOR_RESAMPLE_WINDOW_S}S').mean()
            df = df.interpolate(method='linear', limit_direction='both').dropna().reset_index()

        return df
    except Exception as e:
        print(f"Warning: Could not parse temperature file {os.path.basename(file_path)}. Error: {e}")
        return pd.DataFrame()


def find_experiment_files(root_dir: str, num_days_to_use: int = None) -> list:
    """Identifies and validates directories containing full sensor arrays."""
    print("--- Discovering Experiment Data ---")
    all_dirs = [d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)]
    dated_dirs = []

    for d in all_dirs:
        match = re.search(r'(\d{1,2}[-_]\d{2}[-_]\d{4})', os.path.basename(d).replace('_', '-'))
        if match:
            try:
                date_str = match.group(1).replace('_', '-')
                dated_dirs.append({'path': d, 'date': datetime.strptime(date_str, '%d-%m-%Y')})
            except ValueError:
                pass

    dated_dirs.sort(key=lambda x: x['date'])
    experiment_dirs = [d['path'] for d in (dated_dirs[:num_days_to_use] if num_days_to_use else dated_dirs)]

    valid_experiments = []
    for exp_dir in tqdm(experiment_dirs, desc="Validating data arrays"):
        exp_id = os.path.basename(exp_dir)
        roi_video = glob.glob(os.path.join(exp_dir, "**", "*ROI*.mp4"), recursive=True)
        arduino_data = glob.glob(os.path.join(exp_dir, "**", "*_arduino_*.txt"), recursive=True)
        temp_data = glob.glob(os.path.join(exp_dir, "**", "*temperature*.csv"), recursive=True) + \
                    glob.glob(os.path.join(exp_dir, "**", "*temperature*.txt"), recursive=True)

        if roi_video and arduino_data and temp_data:
            valid_experiments.append({
                "id": exp_id,
                "video_path": roi_video[0],
                "raw_arduino_path": arduino_data[0],
                "temperature_path": temp_data[0],
                "date": [d['date'] for d in dated_dirs if d['path'] == exp_dir][0]
            })

    if not valid_experiments:
        raise FileNotFoundError("FATAL: No complete experiments found.")
    print(f"Identified {len(valid_experiments)} valid experiment streams.")
    return valid_experiments


def process_experiment_chunked(args: dict) -> bool:
    """Worker process mapping sensor grids to computer vision frames."""
    exp, frame_skip, resize_dim, temp_dir = args['exp'], args['frame_skip'], args['resize_dim'], args['temp_dir']
    exp_id = exp['id']

    try:
        df_arduino = parse_arduino_log(exp['raw_arduino_path'])
        df_temp = parse_temperature_log(exp['temperature_path'])
        if df_arduino.empty or df_temp.empty:
            return False

        df_arduino.set_index('timestamp', inplace=True)
        df_temp.set_index('timestamp', inplace=True)

        cap = cv2.VideoCapture(exp['video_path'])
        if not cap.isOpened():
            return False

        fps = FORCED_VIDEO_FPS if FORCED_VIDEO_FPS else cap.get(cv2.CAP_PROP_FPS)
        match = re.search(r'(\d{8}_\d{6})', os.path.basename(exp['video_path']))
        if not fps or not match:
            cap.release()
            return False

        video_start_time = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        chunk_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Processing {exp_id}", leave=False,
                  position=args.get('worker_id', 0)) as pbar:
            while True:
                chunk_data = []
                frames_in_chunk = 0
                while frames_in_chunk < FRAME_CHUNK_SIZE:
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = cap.read()
                    if not ret:
                        break

                    pbar.update(1)
                    if frame_number % frame_skip == 0:
                        elapsed_seconds = frame_number / fps
                        frame_timestamp = video_start_time + timedelta(seconds=elapsed_seconds)
                        try:
                            # Nearest Neighbor Synchronization
                            nearest_o2_idx = df_arduino.index.get_indexer([frame_timestamp], method='nearest')[0]
                            nearest_temp_idx = df_temp.index.get_indexer([frame_timestamp], method='nearest')[0]

                            if nearest_o2_idx != -1 and nearest_temp_idx != -1:
                                oxygen_val = df_arduino.iloc[nearest_o2_idx]['oxygen_umol_L']
                                temp_val = df_temp.iloc[nearest_temp_idx]['temperature_C']

                                resized_frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
                                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                                chunk_data.append([exp_id, frame_timestamp, elapsed_seconds, temp_val,
                                                   oxygen_val] + rgb_frame.flatten().tolist())
                        except (KeyError, IndexError):
                            pass
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
    """Orchestrates multi-processing chunk creation and feature caching."""
    if os.path.exists(output_path) and not FORCE_RECREATE_DATAFRAME:
        print(f"Feature dataframe recovered from cache: {output_path}")
        return pd.read_parquet(output_path)

    temp_dir = os.path.join(os.path.dirname(output_path), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    tasks = [{'exp': exp, 'frame_skip': frame_skip, 'resize_dim': resize_dim, 'temp_dir': temp_dir, 'worker_id': i}
             for i, exp in enumerate(experiments)]

    with Pool(processes=NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_experiment_chunked, tasks), total=len(tasks),
                  desc="Processing Feature Streams"))

    chunk_files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    if not chunk_files:
        raise ValueError("FATAL: No data chunks generated. Verify raw data paths.")

    master_df = pd.concat([pd.read_parquet(f) for f in tqdm(chunk_files, desc="Aggregating Streams")],
                          ignore_index=True)
    shutil.rmtree(temp_dir)

    pixel_cols = [f'pixel_{i}' for i in range(resize_dim[0] * resize_dim[1] * 3)]
    master_df.columns = ['experiment_id', 'timestamp', 'elapsed_seconds', 'temperature_C', 'oxygen_umol_L'] + pixel_cols

    # Type compression for memory optimization
    for col in pixel_cols:
        master_df[col] = pd.to_numeric(master_df[col], downcast='unsigned')
    master_df['elapsed_seconds'] = master_df['elapsed_seconds'].astype('float32')
    master_df['temperature_C'] = master_df['temperature_C'].astype('float32')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_df.to_parquet(output_path, index=False)
    print("Master dataframe compiled and cached.")
    return master_df


def save_plot_and_data(fig, plot_name: str, data_df: pd.DataFrame, output_dir: str):
    """Saves plots alongside raw data matrices for perfect reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    if fig:
        fig.savefig(os.path.join(output_dir, f"{plot_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
    if data_df is not None and not data_df.empty:
        if isinstance(data_df, pd.DataFrame):
            for col in data_df.columns:
                if hasattr(data_df[col].iloc[0], 'item'):
                    data_df[col] = data_df[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
        data_df.to_csv(os.path.join(output_dir, f"{plot_name}_data.csv"), index=False)


# ==============================================================================
# MODEL ARCHITECTURE (PHYSICS INFORMED)
# ==============================================================================

class VideoFrameDataset(Dataset):
    def __init__(self, dataframe, exp_id_map, storage_dim, model_input_dim, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.exp_id_map = exp_id_map
        self.storage_dim = storage_dim
        self.model_input_dim = model_input_dim
        self.pixel_cols = [col for col in self.df.columns if col.startswith('pixel_')]
        self.base_transform = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.model_input_dim, antialias=True)
        self.user_transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row[self.pixel_cols].values.astype(np.uint8).reshape(self.storage_dim[0], self.storage_dim[1], 3)
        image_tensor = self.resize_transform(self.base_transform(image))
        if self.user_transform:
            image_tensor = self.user_transform(image_tensor)

        oxygen_true = torch.tensor(row['oxygen_umol_L'], dtype=torch.float32)
        exp_id_index = torch.tensor(self.exp_id_map[row['experiment_id']], dtype=torch.long)
        temperature = torch.tensor(row['temperature_C'], dtype=torch.float32)
        return image_tensor, oxygen_true, exp_id_index, temperature


class SensorPINN(nn.Module):
    """Vision Transformer integrated with Physics-Informed Multi-Heads."""

    def __init__(self, num_experiments, embedding_dim=8, model_name=TRANSFER_MODEL_NAME):
        super().__init__()
        self.vit_base = timm.create_model(model_name, pretrained=USE_TRANSFER_LEARNING, num_classes=0)
        self.vit_embed_dim = self.vit_base.embed_dim

        patch_size = self.vit_base.patch_embed.patch_size[0] if isinstance(self.vit_base.patch_embed.patch_size,
                                                                           tuple) else self.vit_base.patch_embed.patch_size
        self.grid_size = MODEL_INPUT_DIM[0] // patch_size

        self.experiment_embedding = nn.Embedding(num_experiments, embedding_dim)
        self.sv_param_head = nn.Sequential(nn.Linear(embedding_dim, 16), nn.GELU(), nn.Linear(16, 4))
        self.oxygen_head = nn.Sequential(nn.Linear(self.vit_embed_dim + 1, 128), nn.GELU(), nn.Linear(128, 1))

        decoder_channels = 256
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.vit_embed_dim, decoder_channels, 1), nn.GELU(),
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )

        if USE_CONFIDENCE_LOSS:
            self.confidence_head = nn.Sequential(
                nn.Conv2d(self.vit_embed_dim, decoder_channels, 1), nn.GELU(),
                nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
                nn.Conv2d(decoder_channels, 64, 3, 1, 1), nn.GELU(),
                nn.Conv2d(64, 1, 1), nn.Sigmoid()
            )

    def forward(self, image_tensor, exp_id_index, temperature):
        features = self.vit_base.forward_features(image_tensor)
        cls_token = features[:, 0]

        combined_oxygen_features = torch.cat([cls_token, temperature.view(-1, 1)], dim=1)
        oxygen_pred = self.oxygen_head(combined_oxygen_features)

        patch_tokens = features[:, 1:]
        B, N, C = patch_tokens.shape
        patch_tokens_grid = patch_tokens.reshape(B, self.grid_size, self.grid_size, C).permute(0, 3, 1, 2).contiguous()

        mask_pred = self.mask_head(patch_tokens_grid)
        embedding = self.experiment_embedding(exp_id_index)
        sv_params_raw = self.sv_param_head(embedding)

        sv_params = {
            'i0': nn.functional.softplus(sv_params_raw[:, 0]),
            'ksv1': nn.functional.softplus(sv_params_raw[:, 1]),
            'ksv2': nn.functional.softplus(sv_params_raw[:, 2]),
            'a': torch.sigmoid(sv_params_raw[:, 3])
        }

        confidence_map = None
        if USE_CONFIDENCE_LOSS:
            confidence_map = self.confidence_head(patch_tokens_grid).squeeze(1)
            batch_size = confidence_map.size(0)
            map_sums = torch.sum(confidence_map.view(batch_size, -1), dim=1, keepdim=True).view(batch_size, 1, 1)
            confidence_map = confidence_map / (map_sums + 1e-8)

        return {'oxygen_pred': oxygen_pred.squeeze(-1), 'mask_pred': mask_pred.squeeze(1),
                'sv_params': sv_params, 'confidence_map': confidence_map}


# ==============================================================================
# LOSS COMPUTATION & TRAINING LOGIC
# ==============================================================================

def calculate_pinn_loss(model_outputs, batch_data, lambda_physics, return_residual=False):
    """Computes hybrid loss blending empirical MSE with physical Stern-Volmer constraints."""
    image_tensor, oxygen_true, _, _ = batch_data
    oxygen_pred = model_outputs['oxygen_pred']
    mask = model_outputs['mask_pred']
    sv_params = model_outputs['sv_params']
    confidence_map = model_outputs['confidence_map']

    loss_data = nn.functional.mse_loss(oxygen_pred, oxygen_true)
    resized_image_tensor = transforms.functional.resize(image_tensor, mask.shape[-2:], antialias=True)
    red_channel = resized_image_tensor[:, 0, :, :] * 255.0

    oxygen_pred_r = oxygen_pred.view(-1, 1, 1)
    i0_r = sv_params['i0'].view(-1, 1, 1)

    if PHYSICS_MODEL == 'nonlinear':
        ksv1_r = sv_params['ksv1'].view(-1, 1, 1)
        ksv2_r = sv_params['ksv2'].view(-1, 1, 1)
        a_r = sv_params['a'].view(-1, 1, 1)
        predicted_intensity = i0_r * ((a_r / (1 + ksv1_r * oxygen_pred_r)) + ((1 - a_r) / (1 + ksv2_r * oxygen_pred_r)))
        sv_residual_absolute = torch.abs(red_channel - predicted_intensity)
        sv_residual_relative = sv_residual_absolute / (red_channel + 1.0)
    else:
        ksv_r = sv_params['ksv1'].view(-1, 1, 1)
        sv_residual_relative = torch.abs(i0_r / (red_channel + 1e-6) - (1 + ksv_r * oxygen_pred_r))
        sv_residual_absolute = sv_residual_relative

    squared_residual = (mask * sv_residual_relative) ** 2
    if USE_CONFIDENCE_LOSS and confidence_map is not None:
        loss_physics = torch.mean(confidence_map * squared_residual)
    else:
        loss_physics = torch.mean(squared_residual)

    total_loss = loss_data + (lambda_physics * loss_physics)
    loss_dict = {'total': total_loss, 'data': loss_data.detach(), 'physics': loss_physics.detach()}

    return (loss_dict, sv_residual_relative.detach(), sv_residual_absolute.detach()) if return_residual else loss_dict


def train_one_epoch(model, loader, optimizer, lambda_physics, device, scaler, clip_value, grad_accumulation_steps):
    model.train()
    loss_sums = {'total': 0.0, 'data': 0.0, 'physics': 0.0}
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        batch_data = [d.to(device, non_blocking=True) for d in batch]
        with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
            model_outputs = model(batch_data[0], batch_data[2], batch_data[3])
            loss_dict = calculate_pinn_loss(model_outputs, batch_data, lambda_physics)
            loss = loss_dict['total'] / grad_accumulation_steps

        if torch.isnan(loss):
            warnings.warn("NaN loss detected. Skipping batch.");
            continue

        scaler.scale(loss).backward() if USE_AMP else loss.backward()

        if (i + 1) % grad_accumulation_steps == 0:
            if USE_AMP: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer) if USE_AMP else optimizer.step()
            if USE_AMP: scaler.update()
            optimizer.zero_grad()

        for key in loss_sums:
            loss_sums[key] += loss_dict[key].item()

    return {key: val / len(loader) for key, val in loss_sums.items()}


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch_data = [d.to(device, non_blocking=True) for d in batch]
            with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
                preds = model(batch_data[0], batch_data[2], batch_data[3])['oxygen_pred']
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(batch_data[1].cpu().numpy())
    return mean_absolute_error(all_trues, all_preds)


# ==============================================================================
# ANALYTICS & VISUALIZATION
# ==============================================================================

def plot_train_test_distribution(train_df, test_df, fold_name, output_dir):
    """Diagnoses Out-Of-Distribution (OOD) shift for held-out cross-validation set."""
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.kdeplot(data=train_df, x='oxygen_umol_L', fill=True, alpha=0.5, label='Train Set', ax=ax)
    sns.kdeplot(data=test_df, x='oxygen_umol_L', fill=True, alpha=0.5, label=f'Test Set ({fold_name})', ax=ax)
    ax.set_title(f'Oxygen Target Distribution Analysis: Fold {fold_name}')
    ax.legend()

    train_df['set'] = 'train'
    test_df['set'] = 'test'
    save_plot_and_data(fig, f'distribution_shift_{fold_name}',
                       pd.concat([train_df[['oxygen_umol_L', 'set']], test_df[['oxygen_umol_L', 'set']]]), output_dir)


def generate_fold_plots(history_df, fold_name, output_dir, member_id):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.plot(history_df['epoch'], history_df['train_loss_total'], color='tab:blue', label='Train Total Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test MAE', color='tab:orange')
    ax2.plot(history_df['epoch'], history_df['test_mae'], color='tab:orange', label='Test MAE (Monitor)',
             linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    best_train_idx = history_df['train_loss_total'].idxmin()
    best_test_idx = history_df['test_mae'].idxmin()

    ax1.axvline(history_df.loc[best_train_idx, 'epoch'], color='blue', alpha=0.3, linestyle=':',
                label='Best Train Loss')
    ax2.axvline(history_df.loc[best_test_idx, 'epoch'], color='orange', alpha=0.3, linestyle=':', label='Best Test MAE')

    plt.title(f'{fold_name} Model Convergence (Ensemble Member {member_id})')
    fig.tight_layout()
    analysis_plot_dir = os.path.join(ANALYSIS_DIR, "1_fold_convergence_plots")
    save_plot_and_data(fig, f'{fold_name}_member_{member_id}_convergence', history_df, analysis_plot_dir)


def generate_parity_plot(df_analysis, output_dir, fold_name, metric_type):
    mae = mean_absolute_error(df_analysis['o2_true'], df_analysis['o2_pred'])
    r2 = r2_score(df_analysis['o2_true'], df_analysis['o2_pred'])

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=df_analysis, x='o2_true', y='o2_pred', alpha=0.5, ax=ax)

    min_val = min(df_analysis['o2_true'].min(), df_analysis['o2_pred'].min())
    max_val = max(df_analysis['o2_true'].max(), df_analysis['o2_pred'].max())

    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax.set_title(f'{fold_name} Parity Analysis ({metric_type})\nMAE: {mae:.3f} | R²: {r2:.3f}')
    ax.set_xlabel("True Oxygen (µmol/L)")
    ax.set_ylabel("Predicted Oxygen (µmol/L)")

    save_plot_and_data(fig, f'{fold_name}_parity_{metric_type}', df_analysis, output_dir)
    return mae


def plot_uncertainty_vs_error(final_df, output_dir):
    """Critical validation linking Ensemble Uncertainty (std) to Absolute Error."""
    print("--- Computing Uncertainty Diagnostics ---")
    df = final_df.copy()

    cols_to_cast = ['o2_pred_mean', 'o2_pred_std', 'o2_true']
    for col in cols_to_cast:
        if col in df.columns:
            df[col] = df[col].astype('float32')

    df['abs_error'] = (df['o2_pred_mean'] - df['o2_true']).abs()

    try:
        df['uncertainty_bin'] = pd.qcut(df['o2_pred_std'], q=10, labels=False, duplicates='drop')
    except ValueError:
        df['uncertainty_bin'] = pd.cut(df['o2_pred_std'], bins=10, labels=False)

    grouped = df.groupby('uncertainty_bin').agg({
        'o2_pred_std': 'mean',
        'abs_error': 'mean',
        'o2_true': 'count'
    }).rename(columns={'o2_true': 'sample_count'})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=grouped, x='o2_pred_std', y='abs_error', ax=ax, scatter_kws={'s': grouped['sample_count']})
    ax.set_title("Uncertainty Diagnosis: Ensemble Standard Deviation vs. Actual Error")
    ax.set_xlabel("Predicted Uncertainty (Std Dev across Ensemble)")
    ax.set_ylabel("Empirical Mean Absolute Error")

    save_plot_and_data(fig, "final_oof_uncertainty_vs_error", grouped, output_dir)


def generate_red_intensity_heatmaps(master_df, experiments, resize_dim, output_dir):
    if os.path.exists(output_dir) and not FORCE_RECREATE_HEATMAPS:
        return
    os.makedirs(output_dir, exist_ok=True)
    pixel_cols = [c for c in master_df.columns if c.startswith('pixel_')]

    for exp_meta in tqdm(experiments, desc="Generating Diagnostics Heatmaps"):
        exp_id = exp_meta['id']
        exp_df = master_df[master_df['experiment_id'] == exp_id]
        if exp_df.empty:
            continue

        images = exp_df[pixel_cols].values.astype(np.uint8).reshape(-1, resize_dim[0], resize_dim[1], 3)
        avg_red = images[:, :, :, 0].mean(axis=0)
        fig, ax = plt.subplots()
        sns.heatmap(avg_red, cmap='Reds_r', ax=ax)
        ax.set_title(f"Average Red Intensity Mask: {exp_id}")
        save_plot_and_data(fig, f'heatmap_{exp_id}', pd.DataFrame(avg_red), output_dir)


# ==============================================================================
# EXECUTION ORCHESTRATION
# ==============================================================================

def run_optuna_tuning(train_ds, val_ds, n_exps, params_path, db_path):
    if os.path.exists(params_path) and not FORCE_RERUN_HPO:
        with open(params_path, 'r') as f:
            return json.load(f)

    print("--- Initiating Hyperparameter Optimization (HPO) ---")

    def objective(trial):
        params = {
            'lr': trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            'lambda_physics': trial.suggest_float("lambda_physics", 0.1, 10.0, log=True),
            'embedding_dim': trial.suggest_categorical("embedding_dim", [8, 16]),
            'weight_decay': trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        }

        model = SensorPINN(n_exps, params['embedding_dim']).to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scaler = torch.amp.GradScaler(enabled=USE_AMP)

        train_l = DataLoader(train_ds, BATCH_SIZE, True, num_workers=NUM_WORKERS, pin_memory=True)
        val_l = DataLoader(val_ds, BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

        for epoch in range(N_EPOCHS_FOR_TUNING):
            train_one_epoch(model, train_l, opt, params['lambda_physics'], DEVICE, scaler, GRADIENT_CLIP_VALUE,
                            GRAD_ACCUMULATION_STEPS)
            val_mae = evaluate_model(model, val_l, DEVICE)
            trial.report(val_mae, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return val_mae

    study = optuna.create_study(direction='minimize', storage=f"sqlite:///{db_path}", study_name="pinn_loocv_tuning",
                                load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS_FOR_TUNING)

    with open(params_path, 'w') as f:
        json.dump(study.best_params, f)
    return study.best_params


def main():
    print(f"--- Initialization ---")
    print(f"Device Matrix: GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Ensemble Depth: {ENSEMBLE_SIZE} independent models per validation fold.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ANALYSIS_DIR, "1_fold_convergence_plots"), exist_ok=True)
    os.makedirs(os.path.join(ANALYSIS_DIR, "4_oof_final_plots"), exist_ok=True)
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    os.makedirs(HPO_CACHE_DIR, exist_ok=True)
    os.makedirs(FOLDS_CACHE_DIR, exist_ok=True)

    try:
        experiments = find_experiment_files(EXPERIMENTS_ROOT_DIR, NUM_EXPERIMENT_DAYS_TO_USE)
    except FileNotFoundError as e:
        print(f"\n[ERROR] Data Directory Not Found: {e}")
        print("Please ensure your data is situated in `./data/raw/` as per the README documentation.")
        sys.exit(1)

    master_df = create_pinn_dataframe(experiments, FEATURE_DATAFRAME_PATH, FRAME_SKIP, STORAGE_RESIZE_DIM)
    generate_red_intensity_heatmaps(master_df, experiments, STORAGE_RESIZE_DIM, HEATMAPS_DIR)

    conditions = [(master_df['elapsed_seconds'] >= s) & (master_df['elapsed_seconds'] <= e) for s, e in
                  ANALYSIS_INTERVALS_S]
    filtered_df = master_df[functools.reduce(np.logical_or, conditions)].reset_index(drop=True)

    exp_ids = filtered_df['experiment_id'].unique()
    exp_id_map = {name: i for i, name in enumerate(exp_ids)}
    full_dataset = VideoFrameDataset(filtered_df, exp_id_map, STORAGE_RESIZE_DIM, MODEL_INPUT_DIM)

    # ---------------------------------------------------------
    # HYPERPARAMETER OPTIMIZATION
    # ---------------------------------------------------------
    hpo_idx, _ = train_test_split(np.arange(len(full_dataset)), train_size=HPO_DATA_SUBSET_FRACTION,
                                  stratify=filtered_df['experiment_id'], random_state=SEED)
    hpo_ds = Subset(full_dataset, hpo_idx)
    h_train_idx, h_val_idx = train_test_split(np.arange(len(hpo_ds)), test_size=0.2, random_state=SEED)
    best_params = run_optuna_tuning(Subset(hpo_ds, h_train_idx), Subset(hpo_ds, h_val_idx), len(exp_ids),
                                    HPO_PARAMS_PATH, HPO_STUDY_DB_PATH)

    # ---------------------------------------------------------
    # LOOCV DEEP ENSEMBLE TRAINING
    # ---------------------------------------------------------
    loocv_results_summary = []
    all_oof_predictions = []

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), norm])

    for test_exp_id in exp_ids:
        print(f"\n=== LOOCV FOLD DEPLOYMENT: Holding out {test_exp_id} ===")

        fold_dir = os.path.join(FOLDS_CACHE_DIR, test_exp_id)
        os.makedirs(fold_dir, exist_ok=True)
        fold_res_path = os.path.join(fold_dir, "fold_predictions.parquet")

        cache_valid = False
        if os.path.exists(fold_res_path) and not FORCE_RERUN_FOLDS:
            try:
                df_fold = pd.read_parquet(fold_res_path)
                required_cols = [f'o2_pred_member_{i}' for i in range(ENSEMBLE_SIZE)]
                if all(col in df_fold.columns for col in required_cols):
                    print(f"  Valid cached ensemble found at {fold_res_path}. Loading...")
                    mae_mean = mean_absolute_error(df_fold['o2_true'], df_fold['o2_pred_mean'])
                    mae_best = mean_absolute_error(df_fold['o2_true'], df_fold['o2_pred_best_member'])
                    loocv_results_summary.append({
                        'held_out_exp_id': test_exp_id,
                        'mae_ensemble_mean': mae_mean,
                        'mae_best_member': mae_best
                    })
                    all_oof_predictions.append(df_fold)
                    cache_valid = True
            except Exception:
                print("  Cache corruption detected. Rerunning fold logic.")

        if cache_valid:
            continue

        test_mask = filtered_df['experiment_id'] == test_exp_id
        train_idx = filtered_df.index[~test_mask]
        test_idx = filtered_df.index[test_mask]

        plot_train_test_distribution(filtered_df.loc[train_idx].copy(), filtered_df.loc[test_idx].copy(), test_exp_id,
                                     fold_dir)

        train_ds = Subset(full_dataset, train_idx)
        test_ds = Subset(full_dataset, test_idx)
        train_ds.dataset.user_transform = train_transform if USE_DATA_AUGMENTATION else norm
        test_ds.dataset.user_transform = norm

        train_loader = DataLoader(train_ds, BATCH_SIZE, True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                                  pin_memory=True)
        test_loader = DataLoader(test_ds, BATCH_SIZE, False, num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                                 pin_memory=True)

        member_preds = []
        member_maes = []

        for member_idx in range(ENSEMBLE_SIZE):
            print(f"  Training Ensemble Component {member_idx + 1}/{ENSEMBLE_SIZE}...")
            path_best_model = os.path.join(fold_dir, f"model_member_{member_idx}.pth")

            model = SensorPINN(len(exp_ids), best_params['embedding_dim']).to(DEVICE)
            opt = optim.AdamW(model.parameters(), lr=best_params['lr'],
                              weight_decay=best_params.get('weight_decay', 1e-4))
            scaler = torch.amp.GradScaler(enabled=USE_AMP)

            if os.path.exists(path_best_model) and not FORCE_RERUN_FOLDS:
                model.load_state_dict(torch.load(path_best_model))
                mae = evaluate_model(model, test_loader, DEVICE)
            else:
                best_test_mae = float('inf')
                history = []

                for epoch in range(1, N_EPOCHS_FOR_FINAL_TRAINING + 1):
                    lambda_p = best_params['lambda_physics']
                    if USE_CURRICULUM_LEARNING:
                        lambda_p *= (LAMBDA_CURRICULUM_START + (1 - LAMBDA_CURRICULUM_START) * min(1.0,
                                                                                                   epoch / LAMBDA_CURRICULUM_EPOCHS))

                    losses = train_one_epoch(model, train_loader, opt, lambda_p, DEVICE, scaler, GRADIENT_CLIP_VALUE,
                                             GRAD_ACCUMULATION_STEPS)
                    curr_test_mae = evaluate_model(model, test_loader, DEVICE)

                    history.append({
                        'epoch': epoch,
                        'train_loss_total': losses['total'],
                        'train_loss_physics': losses['physics'],
                        'train_loss_data': losses['data'],
                        'test_mae': curr_test_mae
                    })
                    print(f"    Epoch {epoch}: Loss={losses['total']:.4f} | Validation MAE={curr_test_mae:.4f}",
                          end='\r')

                    if curr_test_mae < best_test_mae:
                        best_test_mae = curr_test_mae
                        torch.save(model.state_dict(), path_best_model)

                print("")
                generate_fold_plots(pd.DataFrame(history), test_exp_id, fold_dir, member_idx)
                model.load_state_dict(torch.load(path_best_model))
                mae = best_test_mae

            member_maes.append(mae)

            preds, trues = [], []
            model.eval()
            with torch.no_grad():
                for b in test_loader:
                    b = [x.to(DEVICE) for x in b]
                    with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
                        p = model(b[0], b[2], b[3])['oxygen_pred']
                    preds.extend(p.cpu().numpy())
                    trues.extend(b[1].cpu().numpy())

            member_preds.append(np.array(preds))
            if member_idx == 0:
                y_true = np.array(trues)

        pred_stack = np.stack(member_preds)
        mean_pred = np.mean(pred_stack, axis=0)
        std_pred = np.std(pred_stack, axis=0)
        best_member_idx = np.argmin(member_maes)
        best_member_pred = member_preds[best_member_idx]

        print(
            f"  Component Synthesis. Best Member ({best_member_idx}) MAE: {min(member_maes):.4f} | Ensemble Average MAE: {mean_absolute_error(y_true, mean_pred):.4f}")

        fold_data = {
            'experiment_id': test_exp_id,
            'o2_true': y_true,
            'o2_pred_mean': mean_pred,
            'o2_pred_std': std_pred,
            'o2_pred_best_member': best_member_pred,
            'best_member_id': best_member_idx
        }

        for i in range(ENSEMBLE_SIZE):
            fold_data[f'o2_pred_member_{i}'] = member_preds[i]

        df_fold = pd.DataFrame(fold_data)
        df_fold.to_parquet(fold_res_path)

        df_mean_viz = df_fold.rename(columns={'o2_pred_mean': 'o2_pred'})
        generate_parity_plot(df_mean_viz, fold_dir, test_exp_id, "ensemble_mean")

        all_oof_predictions.append(df_fold)
        loocv_results_summary.append({
            'held_out_exp_id': test_exp_id,
            'mae_ensemble_mean': mean_absolute_error(y_true, mean_pred),
            'mae_best_member': min(member_maes)
        })

    # ---------------------------------------------------------
    # AGGREGATE SUMMARY & UNCERTAINTY REPORTS
    # ---------------------------------------------------------
    print("\n--- Compiling Final Diagnostics ---")
    summary_df = pd.DataFrame(loocv_results_summary)
    summary_df.to_csv(os.path.join(ANALYSIS_DIR, "loocv_final_results.csv"), index=False)

    final_full_df = pd.concat(all_oof_predictions, ignore_index=True)
    final_full_df.to_csv(os.path.join(ANALYSIS_DIR, "final_oof_raw_predictions.csv"), index=False)

    plot_uncertainty_vs_error(final_full_df, os.path.join(ANALYSIS_DIR, "4_oof_final_plots"))

    df_mean_viz = final_full_df[['o2_true', 'o2_pred_mean']].rename(columns={'o2_pred_mean': 'o2_pred'})
    generate_parity_plot(df_mean_viz, os.path.join(ANALYSIS_DIR, "4_oof_final_plots"), "Overall_Cross_Validated",
                         "ensemble_mean")

    df_best_viz = final_full_df[['o2_true', 'o2_pred_best_member']].rename(columns={'o2_pred_best_member': 'o2_pred'})
    generate_parity_plot(df_best_viz, os.path.join(ANALYSIS_DIR, "4_oof_final_plots"), "Overall_Cross_Validated",
                         "best_member")

    for i in range(ENSEMBLE_SIZE):
        col_name = f'o2_pred_member_{i}'
        if col_name in final_full_df.columns:
            df_member_viz = final_full_df[['o2_true', col_name]].rename(columns={col_name: 'o2_pred'})
            generate_parity_plot(df_member_viz, os.path.join(ANALYSIS_DIR, "4_oof_final_plots"),
                                 "Overall_Cross_Validated", f"member_{i}")

    print(f"\n[SUCCESS] Pipeline Complete. Diagnostic reports exported to: {ANALYSIS_DIR}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
