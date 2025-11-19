"""
Large-Scale Ensemble Execution Time Predictor with Deep Neural Networks

Architecture Enhancements:
- MUCH deeper networks: [512, 512, 256, 256, 128, 128, 64, 64, 32]
- Residual connections (skip connections) for gradient flow
- Layer normalization instead of batch norm for better stability
- LeakyReLU and GELU activations
- Multi-head attention mechanism for feature interactions
- Separate feature embedding layers
- Ensemble of 5 models (fast/medium-fast/medium/medium-slow/slow)
- Advanced regularization: dropout, weight decay, gradient clipping

Advanced Training Techniques (NEW):
- Data Augmentation: Synthetic sample generation via interpolation
- Transfer Learning: Pre-train on all data, fine-tune per regime
- Multi-Task Learning: Shared feature extractor with regime-specific heads
"""

import pandas as pd
import numpy as np
import pickle
import math
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class KernelDataset(Dataset):
    """PyTorch dataset for kernel execution time prediction"""

    def __init__(self, X, y, kernel_ids=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        self.kernel_ids = kernel_ids if kernel_ids is not None else np.zeros(len(X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def augment_data(X, y, kernel_ids, augmentation_factor=3, seed=42):
    """
    Generate synthetic samples via kernel-aware interpolation

    Args:
        X: Feature matrix (N x D)
        y: Target values (log exec time)
        kernel_ids: Kernel IDs to ensure same-kernel mixing
        augmentation_factor: Number of synthetic samples per real sample

    Returns:
        X_augmented, y_augmented, kernel_ids_augmented
    """
    np.random.seed(seed)
    X_aug_list, y_aug_list, kid_aug_list = [X], [y], [kernel_ids]

    # Group by kernel
    unique_kernels = np.unique(kernel_ids)

    for kernel_id in unique_kernels:
        mask = kernel_ids == kernel_id
        X_kernel = X[mask]
        y_kernel = y[mask]

        if len(X_kernel) < 2:
            continue  # Need at least 2 samples to interpolate

        n_samples = len(X_kernel)
        n_synthetic = min(n_samples * augmentation_factor, n_samples * 5)  # Cap at 5x

        for _ in range(n_synthetic):
            # Pick two random samples from same kernel
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)

            # Beta distribution favors balanced mixing (0.5)
            alpha = np.random.beta(2, 2)

            # Interpolate features (some features should not be interpolated)
            X_new = alpha * X_kernel[idx1] + (1 - alpha) * X_kernel[idx2]

            # For discrete features (uses_*, control_flow_ops, etc.), use majority vote
            # Assume first feature is kernel_encoded, keep it fixed
            X_new[0] = X_kernel[idx1, 0]

            # Interpolate log(time) - more stable
            y_new = alpha * y_kernel[idx1] + (1 - alpha) * y_kernel[idx2]

            X_aug_list.append(X_new.reshape(1, -1))
            y_aug_list.append([y_new])
            kid_aug_list.append([kernel_id])

    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.concatenate(y_aug_list)
    kernel_ids_augmented = np.concatenate(kid_aug_list)

    return X_augmented, y_augmented, kernel_ids_augmented


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention for feature interactions"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.reshape(batch_size, self.dim)
        out = self.fc_out(out)

        return out


class MultiTaskExecTimePredictor(nn.Module):
    """
    Multi-task learning model with shared feature extractor
    and regime-specific prediction heads
    """

    def __init__(
        self,
        input_dim,
        shared_dims=[512, 256, 128],
        head_dims=[64, 32],
        num_regimes=5,
        dropout=0.15,
    ):
        super(MultiTaskExecTimePredictor, self).__init__()

        # Shared feature extractor (learns from ALL data)
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in shared_dims:
            shared_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.shared_encoder = nn.Sequential(*shared_layers)

        # Regime-specific prediction heads
        self.regime_heads = nn.ModuleList()
        for _ in range(num_regimes):
            head_layers = []
            head_prev_dim = prev_dim

            for head_dim in head_dims:
                head_layers.extend(
                    [
                        nn.Linear(head_prev_dim, head_dim),
                        nn.LayerNorm(head_dim),
                        nn.GELU(),
                        nn.Dropout(dropout / 2),
                    ]
                )
                head_prev_dim = head_dim

            head_layers.append(nn.Linear(head_prev_dim, 1))
            self.regime_heads.append(nn.Sequential(*head_layers))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, regime_idx=None):
        # Shared encoding
        shared_features = self.shared_encoder(x)

        # If regime specified, use that head
        if regime_idx is not None:
            return self.regime_heads[regime_idx](shared_features)

        # Otherwise return all predictions (for multi-task training)
        return [head(shared_features) for head in self.regime_heads]


class LargeExecTimePredictor(nn.Module):
    """
    Large-scale deep neural network with advanced architecture

    Features:
    - Feature embedding layers
    - Multi-head attention
    - Residual blocks
    - Deep architecture: [512, 512, 256, 256, 128, 128, 64, 64, 32]
    - Layer normalization
    - GELU and LeakyReLU activations
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 512, 256, 256, 128, 128, 64, 64, 32],
        dropout=0.15,
        use_attention=True,
        num_residual_blocks=3,
    ):
        super(LargeExecTimePredictor, self).__init__()

        self.use_attention = use_attention

        # Feature embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-head attention for feature interactions
        if use_attention:
            self.attention = MultiHeadAttention(
                hidden_dims[0], num_heads=8, dropout=dropout
            )

        # Deep network with residual connections
        layers = []
        prev_dim = hidden_dims[0]

        for i, hidden_dim in enumerate(hidden_dims):
            if prev_dim == hidden_dim and i > 0:
                # Add residual block when dimensions match
                layers.append(ResidualBlock(hidden_dim, dropout=dropout))
            else:
                # Regular layer with dimension change
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU() if i % 2 == 0 else nn.LeakyReLU(0.2),
                        nn.Dropout(dropout),
                    ]
                )
            prev_dim = hidden_dim

        # Additional residual blocks at the end for better representation
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(prev_dim, dropout=dropout))

        self.network = nn.Sequential(*layers)

        # Output head with multiple layers
        self.output_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.LayerNorm(prev_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(prev_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Embed input features
        x = self.input_embed(x)

        # Apply attention
        if self.use_attention:
            x = x + self.attention(x)  # Residual connection

        # Deep network
        x = self.network(x)

        # Output
        x = self.output_head(x)

        return x


class AdvancedModelTrainer:
    """Advanced training with learning rate scheduling and gradient accumulation"""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "epoch": [], "lr": []}

    def train_epoch(
        self, train_loader, criterion, optimizer, gradient_accumulation_steps=1
    ):
        self.model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)

            # Gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader,
        val_loader,
        epochs=300,
        lr=0.0005,
        patience=40,
        warmup_epochs=10,
        verbose=True,
        gradient_accumulation_steps=1,
    ):
        """
        Advanced training loop with:
        - Warmup learning rate
        - Cosine annealing with restarts
        - Early stopping
        - Gradient accumulation
        """
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)
        )

        # Learning rate scheduler: warmup + cosine annealing
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        if verbose:
            print(
                f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<15} {'Status'}"
            )
            print("-" * 70)

        for epoch in range(epochs):
            train_loss = self.train_epoch(
                train_loader, criterion, optimizer, gradient_accumulation_steps
            )
            val_loss = self.validate(val_loader, criterion)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                status = "✓ Best"
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} "
                    f"{current_lr:<15.6e} {status}"
                )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if verbose:
            print(f"\nBest validation loss: {best_val_loss:.6f}")
            print(f"Total epochs: {len(self.history['epoch'])}\n")

        return best_val_loss


def pretrain_base_model(csv_file, output_dir, epochs=100):
    """
    Pre-train a base model on ALL data (transfer learning step 1)
    This model learns general CUDA performance patterns
    """
    print("\n" + "=" * 70)
    print("PRE-TRAINING BASE MODEL ON ALL DATA")
    print("=" * 70)

    # Load all data (no time filtering)
    X, y, features, kernel_encoder, df = prepare_data(
        csv_file, time_min=None, time_max=None
    )

    print(f"Pre-training on {len(X):,} samples")

    # Split and scale
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Datasets
    train_dataset = KernelDataset(X_train_scaled, y_train)
    val_dataset = KernelDataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    # Create base model (slightly smaller for generalization)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    base_model = LargeExecTimePredictor(
        input_dim=len(features),
        hidden_dims=[512, 256, 128, 64, 32],  # Shallower for pre-training
        dropout=0.2,
        use_attention=True,
        num_residual_blocks=2,
    )

    print(f"Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    # Train
    trainer = AdvancedModelTrainer(base_model, device=device)
    best_loss = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.001,
        patience=20,
        warmup_epochs=5,
        gradient_accumulation_steps=1,
    )

    # Save base model
    output_path = Path(output_dir) / "base_pretrained"
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(base_model.state_dict(), output_path / "model.pth")

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "features": features,
                "kernel_encoder": kernel_encoder,
                "best_loss": best_loss,
            },
            f,
        )

    print(f"✓ Pre-trained base model saved to {output_path}/")
    print(f"Best validation loss: {best_loss:.6f}\n")

    return base_model, scaler, features, kernel_encoder


def prepare_features(df):
    """Prepare features with engineering"""
    # N-based features
    df["log_N"] = np.log(df["N"] + 1)
    df["sqrt_N"] = np.sqrt(df["N"])
    df["N_squared"] = df["N"] ** 2
    df["N_cubed"] = df["N"] ** 3

    # Block configuration features
    df["total_threads"] = df["block_x"] * df["block_y"]
    df["log_threads"] = np.log(df["total_threads"] + 1)
    df["block_aspect_ratio"] = df["block_x"] / (df["block_y"] + 1)
    df["block_product"] = df["block_x"] * df["block_y"]

    # Interaction features
    df["N_per_thread"] = df["N"] / (df["total_threads"] + 1)
    df["compute_per_thread"] = (
        df["compute_intensity"] * df["arithmetic_ops"] / (df["total_threads"] + 1)
    )
    df["memory_per_thread"] = df["estimated_memory_bytes"] / (df["total_threads"] + 1)
    df["flops_per_thread"] = df["estimated_flops"] / (df["total_threads"] + 1)

    # Advanced interaction features
    df["compute_to_memory_ratio"] = df["arithmetic_ops"] / (df["memory_ops"] + 1)
    df["read_write_ratio"] = df["global_reads"] / (df["global_writes"] + 1)

    return df


def prepare_data(csv_file, time_min=None, time_max=None, use_kernel_names=True):
    """Prepare data with time filtering"""
    df = pd.read_csv(csv_file)

    # Handle column naming
    column_mapping = {
        "kernel": "kernel_name",
        "bx": "block_x",
        "by": "block_y",
        "bz": "block_z",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Filter by execution time
    if time_min is not None:
        df = df[df["exec_time"] >= time_min]
    if time_max is not None:
        df = df[df["exec_time"] < time_max]

    print(f"Data: {len(df):,} samples")
    print(
        f"Exec time range: {df['exec_time'].min():.6f} - {df['exec_time'].max():.6f} ms"
    )
    print(
        f"Log exec time range: {np.log(df['exec_time'].min()):.2f} - {np.log(df['exec_time'].max()):.2f}"
    )

    # Feature engineering
    df = prepare_features(df)

    # Fill missing new features with 0 (backward compatibility)
    new_features = [
        "control_flow_ops",
        "loop_ops",
        "uses_syncthreads",
        "estimated_flops",
        "estimated_memory_bytes",
        "uses_threadIdx_x",
        "uses_threadIdx_y",
        "uses_threadIdx_z",
        "uses_blockIdx_x",
        "uses_blockIdx_y",
        "uses_blockIdx_z",
        "uses_blockDim_x",
        "uses_blockDim_y",
        "uses_blockDim_z",
    ]
    for feat in new_features:
        if feat not in df.columns:
            df[feat] = 0

    # Define feature columns (expanded)
    numeric_features = [
        "N",
        "log_N",
        "sqrt_N",
        "N_squared",
        "N_cubed",
        "block_x",
        "block_y",
        "total_threads",
        "log_threads",
        "block_aspect_ratio",
        "block_product",
        "N_per_thread",
        "dimensionality",
        "compute_intensity",
        "compute_per_thread",
        "memory_per_thread",
        "flops_per_thread",
        "compute_to_memory_ratio",
        "read_write_ratio",
        "has_shared_memory",
        "global_reads",
        "global_writes",
        "arithmetic_ops",
        "memory_ops",
        "control_flow_ops",
        "loop_ops",
        "uses_syncthreads",
        "estimated_flops",
        "estimated_memory_bytes",
        "uses_threadIdx_x",
        "uses_threadIdx_y",
        "uses_threadIdx_z",
        "uses_blockIdx_x",
        "uses_blockIdx_y",
        "uses_blockIdx_z",
        "uses_blockDim_x",
        "uses_blockDim_y",
        "uses_blockDim_z",
    ]

    # Kernel encoding
    kernel_encoder = None
    if use_kernel_names:
        kernel_encoder = LabelEncoder()
        df["kernel_encoded"] = kernel_encoder.fit_transform(df["kernel_name"])
        numeric_features.insert(0, "kernel_encoded")

    X = df[numeric_features].values
    y = np.log(df["exec_time"].values)  # Always use log

    print(f"Features ({len(numeric_features)}): {numeric_features[:5]}...")

    return X, y, numeric_features, kernel_encoder, df


def evaluate_model(model, test_loader, scaler, device="cpu"):
    """Comprehensive evaluation"""
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).cpu().numpy().flatten()
            all_predictions.append(outputs)
            all_actuals.append(batch_y.numpy().flatten())

    predictions_log = np.concatenate(all_predictions)
    actuals_log = np.concatenate(all_actuals)

    # Convert to real space
    predictions_real = np.exp(predictions_log)
    actuals_real = np.exp(actuals_log)

    # Metrics
    r2_log = r2_score(actuals_log, predictions_log)
    mae_log = mean_absolute_error(actuals_log, predictions_log)

    r2_real = r2_score(actuals_real, predictions_real)
    mae_real = mean_absolute_error(actuals_real, predictions_real)

    mape = np.mean(np.abs((actuals_real - predictions_real) / actuals_real)) * 100

    pct_errors = np.abs((actuals_real - predictions_real) / actuals_real) * 100
    within_10 = (pct_errors < 10).mean() * 100
    within_20 = (pct_errors < 20).mean() * 100

    print(f"  R² (log): {r2_log:.4f}  |  R² (real): {r2_real:.4f}")
    print(f"  MAE (log): {mae_log:.4f}  |  MAE (real): {mae_real:.6f} ms")
    print(
        f"  MAPE: {mape:.2f}%  |  Within 10%: {within_10:.1f}%  |  Within 20%: {within_20:.1f}%"
    )

    return {
        "r2_log": r2_log,
        "r2_real": r2_real,
        "mae_log": mae_log,
        "mae_real": mae_real,
        "mape": mape,
        "within_10pct": within_10,
        "within_20pct": within_20,
    }


def train_specialized_model(
    csv_file,
    time_min,
    time_max,
    model_name,
    output_dir,
    epochs=300,
    use_augmentation=True,
    pretrained_model=None,
    use_transfer_learning=True,
):
    """
    Train a large-scale model for a specific time regime

    Args:
        csv_file: Path to data CSV
        time_min, time_max: Time range for this regime
        model_name: Name for saving
        output_dir: Output directory
        epochs: Training epochs
        use_augmentation: Enable data augmentation (recommended for sparse regimes)
        pretrained_model: Pre-trained base model for transfer learning
        use_transfer_learning: Whether to use transfer learning
    """
    print("\n" + "=" * 70)
    print(f"TRAINING LARGE {model_name.upper()} MODEL")
    print(
        f"Time range: {time_min if time_min else 0:.3f} - {time_max if time_max else 'inf'} ms"
    )
    print(f"Augmentation: {'ON' if use_augmentation else 'OFF'}")
    print(
        f"Transfer Learning: {'ON' if use_transfer_learning and pretrained_model else 'OFF'}"
    )
    print("=" * 70)

    X, y, features, kernel_encoder, df = prepare_data(csv_file, time_min, time_max)
    kernel_ids = df["kernel_encoded"].values

    original_size = len(X)
    print(f"Original samples: {original_size:,}")

    # Data Augmentation for sparse regimes
    if use_augmentation and original_size < 2000:
        augmentation_factor = max(1, min(5, 2000 // original_size))
        print(f"Applying augmentation (factor={augmentation_factor})...")
        X, y, kernel_ids = augment_data(
            X, y, kernel_ids, augmentation_factor=augmentation_factor
        )
        print(f"Augmented samples: {len(X):,} (+{len(X) - original_size:,})")

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Datasets with larger batch size for large model
    train_dataset = KernelDataset(X_train_scaled, y_train)
    test_dataset = KernelDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Large model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transfer Learning: Start from pre-trained model
    if use_transfer_learning and pretrained_model is not None:
        print("Initializing from pre-trained base model...")
        model = LargeExecTimePredictor(
            input_dim=len(features),
            hidden_dims=[512, 512, 256, 256, 128, 128, 64, 64, 32],
            dropout=0.15,
            use_attention=True,
            num_residual_blocks=3,
        )

        # Load pre-trained weights (partial loading for matching layers)
        try:
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()

            # Filter out layers that don't match
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(
                f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pre-trained model"
            )
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
    else:
        model = LargeExecTimePredictor(
            input_dim=len(features),
            hidden_dims=[512, 512, 256, 256, 128, 128, 64, 64, 32],
            dropout=0.15,
            use_attention=True,
            num_residual_blocks=3,
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fine-tuning with lower learning rate if using transfer learning
    lr = 0.0001 if (use_transfer_learning and pretrained_model) else 0.0005

    trainer = AdvancedModelTrainer(model, device=device)
    best_loss = trainer.train(
        train_loader,
        test_loader,
        epochs=epochs,
        lr=lr,
        patience=40,
        warmup_epochs=10,
        gradient_accumulation_steps=2,
    )

    # Evaluate
    print(f"\n{model_name.upper()} MODEL EVALUATION:")
    metrics = evaluate_model(model, test_loader, scaler, device)

    # Save
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pth")

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "features": features,
                "kernel_encoder": kernel_encoder,
                "time_range": (time_min, time_max),
                "metrics": metrics,
                "model_config": {
                    "hidden_dims": [512, 512, 256, 256, 128, 128, 64, 64, 32],
                    "dropout": 0.15,
                    "use_attention": True,
                    "num_residual_blocks": 3,
                },
            },
            f,
        )

    print(f"✓ Saved to {output_path}/")

    return metrics


class EnsemblePredictor:
    """Smart ensemble predictor with 5-regime model routing"""

    def __init__(self, models_dir):
        self.models = {}
        self.metadata = {}

        for model_name in ["fast", "medium_fast", "medium", "medium_slow", "slow"]:
            model_path = Path(models_dir) / model_name
            if model_path.exists():
                # Load metadata
                with open(model_path / "metadata.pkl", "rb") as f:
                    meta = pickle.load(f)

                # Load model
                model = LargeExecTimePredictor(
                    input_dim=len(meta["features"]),
                    **meta.get(
                        "model_config",
                        {
                            "hidden_dims": [512, 512, 256, 256, 128, 128, 64, 64, 32],
                            "dropout": 0.15,
                            "use_attention": True,
                            "num_residual_blocks": 3,
                        },
                    ),
                )
                model.load_state_dict(
                    torch.load(model_path / "model.pth", weights_only=True)
                )
                model.eval()

                self.models[model_name] = model
                self.metadata[model_name] = meta

                print(
                    f"✓ Loaded {model_name} model (R²={meta['metrics']['r2_real']:.3f})"
                )

    def estimate_regime(self, kernel_info):
        """Estimate which time regime this kernel falls into (5 regimes)"""
        # Simple heuristic based on problem size and complexity
        N = kernel_info.get("N", 1024)
        dim = kernel_info.get("dimensionality", 1)
        compute = kernel_info.get("compute_intensity", 1.0)

        # Rough estimate: time ~ N^dim * compute / 100000
        estimated_time = (N**dim) * compute / 100000.0

        if estimated_time < 0.5:
            return "fast"
        elif estimated_time < 5.0:
            return "medium_fast"
        elif estimated_time < 50.0:
            return "medium"
        elif estimated_time < 500.0:
            return "medium_slow"
        else:
            return "slow"

    def predict_single(self, model_name, kernel_info, block_x, block_y):
        """Predict using a specific model"""
        meta = self.metadata[model_name]
        model = self.models[model_name]

        # Prepare features (match training)
        feature_values = []

        # Create temp dataframe to use prepare_features
        temp_df = pd.DataFrame(
            [
                {
                    "N": kernel_info.get("N", 1024),
                    "block_x": block_x,
                    "block_y": block_y,
                    "dimensionality": kernel_info.get("dimensionality", 1),
                    "compute_intensity": kernel_info.get("compute_intensity", 1.0),
                    "has_shared_memory": kernel_info.get("has_shared_memory", 0),
                    "global_reads": kernel_info.get("global_reads", 10),
                    "global_writes": kernel_info.get("global_writes", 5),
                    "arithmetic_ops": kernel_info.get("arithmetic_ops", 100),
                    "memory_ops": kernel_info.get("memory_ops", 15),
                    "control_flow_ops": kernel_info.get("control_flow_ops", 0),
                    "loop_ops": kernel_info.get("loop_ops", 0),
                    "uses_syncthreads": kernel_info.get("uses_syncthreads", 0),
                    "estimated_flops": kernel_info.get("estimated_flops", 0),
                    "estimated_memory_bytes": kernel_info.get(
                        "estimated_memory_bytes", 0
                    ),
                    "uses_threadIdx_x": kernel_info.get("uses_threadIdx_x", 0),
                    "uses_threadIdx_y": kernel_info.get("uses_threadIdx_y", 0),
                    "uses_threadIdx_z": kernel_info.get("uses_threadIdx_z", 0),
                    "uses_blockIdx_x": kernel_info.get("uses_blockIdx_x", 0),
                    "uses_blockIdx_y": kernel_info.get("uses_blockIdx_y", 0),
                    "uses_blockIdx_z": kernel_info.get("uses_blockIdx_z", 0),
                    "uses_blockDim_x": kernel_info.get("uses_blockDim_x", 0),
                    "uses_blockDim_y": kernel_info.get("uses_blockDim_y", 0),
                    "uses_blockDim_z": kernel_info.get("uses_blockDim_z", 0),
                }
            ]
        )

        temp_df = prepare_features(temp_df)

        # Extract feature values in correct order
        for feat in meta["features"]:
            if feat == "kernel_encoded":
                try:
                    encoded = meta["kernel_encoder"].transform(
                        [kernel_info["kernel_name"]]
                    )[0]
                except:
                    encoded = 0
                feature_values.append(encoded)
            elif feat in temp_df.columns:
                feature_values.append(temp_df[feat].iloc[0])
            else:
                feature_values.append(0)

        X = np.array([feature_values])
        X_scaled = meta["scaler"].transform(X)

        # Predict
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            pred_log = model(X_tensor).item()

        return np.exp(pred_log)

    def predict_optimal_config(self, kernel_info, candidate_configs=None):
        """Find optimal config using appropriate model(s)"""
        # Determine regime
        regime = self.estimate_regime(kernel_info)

        if regime not in self.models:
            # Fallback to any available model
            regime = list(self.models.keys())[0]
            print(f"⚠ Using fallback model: {regime}")

        # Generate candidates
        if candidate_configs is None:
            dim = kernel_info.get("dimensionality", 1)
            if dim == 1:
                candidate_configs = [
                    (32, 1),
                    (64, 1),
                    (128, 1),
                    (256, 1),
                    (512, 1),
                    (1024, 1),
                ]
            else:
                candidate_configs = [
                    (8, 8),
                    (16, 8),
                    (16, 16),
                    (32, 8),
                    (32, 16),
                    (32, 32),
                    (64, 4),
                    (64, 8),
                    (64, 16),
                    (128, 4),
                    (128, 8),
                    (256, 4),
                ]

        # Predict for all configs
        predictions = []
        for block_x, block_y in candidate_configs:
            pred_time = self.predict_single(regime, kernel_info, block_x, block_y)
            predictions.append(
                {"block_x": block_x, "block_y": block_y, "predicted_time": pred_time}
            )

        # Find best
        best = min(predictions, key=lambda x: x["predicted_time"])

        return best["block_x"], best["block_y"], best["predicted_time"], predictions


def grid_from_block_and_problem(block_x, block_y, problem_shape, dimensionality=1):
    """Return (grid_x, grid_y, grid_z) that cover problem_shape with block dims."""
    if dimensionality == 1:
        N = int(problem_shape)
        gx = max(1, math.ceil(N / block_x))
        return (gx, 1, 1)
    elif dimensionality == 2:
        Nx, Ny = problem_shape
        gx = max(1, math.ceil(Nx / block_x))
        gy = max(1, math.ceil(Ny / block_y))
        return (gx, gy, 1)
    elif dimensionality == 3:
        Nx, Ny, Nz = problem_shape
        gz = max(1, math.ceil(Nz / 1))
        return (max(1, math.ceil(Nx / block_x)), max(1, math.ceil(Ny / block_y)), gz)
    return (1, 1, 1)


def grid_for_saturation(block_x, block_y, device_profile=None, target_blocks_per_sm=8):
    """
    Create a grid intended to saturate the device if problem size unknown.
    device_profile example: {'sm_count': 80, 'max_grid_x': 2**31-1, 'max_grid_y': 65535}
    """
    if device_profile is None:
        device_profile = {"sm_count": 80, "max_grid_x": 2**31 - 1, "max_grid_y": 65535}

    sm = int(device_profile.get("sm_count", 80))
    max_gx = int(device_profile.get("max_grid_x", 2**31 - 1))
    max_gy = int(device_profile.get("max_grid_y", 65535))

    target_blocks = sm * int(target_blocks_per_sm)
    for gy in (1, 2, 4, 8):
        gx = math.ceil(target_blocks / gy)
        if gx <= max_gx and gy <= max_gy:
            return (min(gx, max_gx), min(gy, max_gy), 1)
    return (sm * target_blocks_per_sm, 1, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Large-Scale Ensemble Execution Time Predictor"
    )
    parser.add_argument("csv_file", nargs="?", help="Path to enriched CSV")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--predict", action="store_true", help="Predict execution time")
    parser.add_argument("--cuda-file", help="Path to CUDA file for prediction")
    parser.add_argument("--N", type=int, help="Problem size N for prediction")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output-dir", default="ensemble_models_large")

    args = parser.parse_args()

    if args.train:
        if not args.csv_file:
            parser.error("CSV file required for training")

        print("=" * 70)
        print("LARGE-SCALE ENSEMBLE PREDICTOR - TRAINING")
        print("=" * 70)
        print("Architecture: Deep ResNet with Multi-Head Attention")
        print("Hidden dims: [512, 512, 256, 256, 128, 128, 64, 64, 32]")
        print("Features: Residual blocks, Layer norm, GELU/LeakyReLU")
        print("\nAdvanced Training Techniques:")
        print("  ✓ Data Augmentation (kernel-aware interpolation)")
        print("  ✓ Transfer Learning (pre-train + fine-tune)")
        print("  ✓ Adaptive augmentation based on sample count")
        print("=" * 70)

        # Step 1: Pre-train base model on all data
        print("\n[STEP 1/6] PRE-TRAINING BASE MODEL")
        base_model, base_scaler, base_features, base_encoder = pretrain_base_model(
            args.csv_file, args.output_dir, epochs=100
        )
        print("\n")

        # Train 5 specialized models with transfer learning
        print("[STEP 2/6] FAST MODEL (<0.5ms)")
        metrics_fast = train_specialized_model(
            args.csv_file,
            time_min=None,
            time_max=0.5,
            model_name="fast",
            output_dir=args.output_dir,
            epochs=args.epochs,
            use_augmentation=True,
            pretrained_model=base_model,
        )

        print("[STEP 3/6] MEDIUM-FAST MODEL (0.5-5ms)")
        metrics_medium_fast = train_specialized_model(
            args.csv_file,
            time_min=0.5,
            time_max=5.0,
            model_name="medium_fast",
            output_dir=args.output_dir,
            epochs=args.epochs,
            use_augmentation=True,
            pretrained_model=base_model,
        )

        print("[STEP 4/6] MEDIUM MODEL (5-50ms)")
        metrics_medium = train_specialized_model(
            args.csv_file,
            time_min=5.0,
            time_max=50.0,
            model_name="medium",
            output_dir=args.output_dir,
            epochs=args.epochs,
            use_augmentation=True,
            pretrained_model=base_model,
        )

        print("[STEP 5/6] MEDIUM-SLOW MODEL (50-500ms)")
        metrics_medium_slow = train_specialized_model(
            args.csv_file,
            time_min=50.0,
            time_max=500.0,
            model_name="medium_slow",
            output_dir=args.output_dir,
            epochs=args.epochs,
            use_augmentation=True,
            pretrained_model=base_model,
        )

        print("[STEP 6/6] SLOW MODEL (>500ms)")
        metrics_slow = train_specialized_model(
            args.csv_file,
            time_min=500.0,
            time_max=None,
            model_name="slow",
            output_dir=args.output_dir,
            epochs=args.epochs,
            use_augmentation=True,
            pretrained_model=base_model,
        )

        # Summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - SUMMARY")
        print("=" * 70)
        print(
            f"Fast (<0.5ms):         R²={metrics_fast['r2_real']:.3f}, MAPE={metrics_fast['mape']:.1f}%"
        )
        print(
            f"Medium-Fast (0.5-5ms): R²={metrics_medium_fast['r2_real']:.3f}, MAPE={metrics_medium_fast['mape']:.1f}%"
        )
        print(
            f"Medium (5-50ms):       R²={metrics_medium['r2_real']:.3f}, MAPE={metrics_medium['mape']:.1f}%"
        )
        print(
            f"Medium-Slow (50-500ms):R²={metrics_medium_slow['r2_real']:.3f}, MAPE={metrics_medium_slow['mape']:.1f}%"
        )
        print(
            f"Slow (>500ms):         R²={metrics_slow['r2_real']:.3f}, MAPE={metrics_slow['mape']:.1f}%"
        )

        print(f"\nModels saved to: {args.output_dir}/")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
