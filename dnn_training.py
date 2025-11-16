#!/usr/bin/env python3
"""
Complete Neural Network for CUDA Block Size Prediction
Works with KLARAPTOR enriched data format
Predicts optimal (block_x, block_y, total_threads) directly
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class KernelDataset(Dataset):
    """PyTorch dataset for kernel configurations"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BlockPredictor(nn.Module):
    """
    Neural network that predicts optimal block dimensions
    
    Architecture:
    - Input: kernel features (N, compute_intensity, etc.)
    - Hidden layers with BatchNorm and Dropout
    - Output: [log2(block_x), log2(block_y), log2(total_threads)]
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3):
        super(BlockPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer: 3 values (block_x, block_y, total_threads in log2 space)
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class BlockPredictorTrainer:
    """Handles training, evaluation, and prediction"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate model"""
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
    
    def train(self, train_loader, val_loader, epochs=200, lr=0.001, 
              patience=25, verbose=True):
        """
        Full training loop with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print progress
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if verbose:
            print("Training Neural Network...")
            print("=" * 70)
            print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<15}")
            print("-" * 70)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {current_lr:<15.6e}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        if verbose:
            print("=" * 70)
            print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def predict(self, X):
        """
        Predict block dimensions for given inputs
        
        Args:
            X: numpy array of features
            
        Returns:
            numpy array of predictions in log2 space
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def save(self, model_path, scaler_path, metadata):
        """Save model, scaler, and metadata"""
        torch.save(self.model.state_dict(), model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': metadata['scaler'],
                'features': metadata['features'],
                'kernel_encoder': metadata.get('kernel_encoder'),
                'history': self.history
            }, f)
        
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")


def prepare_data(csv_file, use_kernel_names=True):
    """
    Prepare data from KLARAPTOR CSV (handles both enriched and full format)
    
    Args:
        csv_file: Path to CSV file (can be enriched or full extraction)
        use_kernel_names: Include kernel name as one-hot encoded feature
        
    Returns:
        X, y, feature_names, scaler, kernel_encoder
    """
    df = pd.read_csv(csv_file)
    
    # Handle different column naming conventions
    column_mapping = {
        'kernel': 'kernel_name',
        'bx': 'block_x',
        'by': 'block_y',
        'bz': 'block_z'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    print(f"Loaded data: {len(df)} rows")
    print(f"Kernels: {df['kernel_name'].nunique()}")
    print(f"Data sizes: {sorted(df['N'].unique())}")
    
    # Check if we have GPU column
    if 'gpu' in df.columns:
        print(f"GPUs: {df['gpu'].nunique()} ({', '.join(df['gpu'].unique())})")
    
    # Find optimal configuration for each (kernel, N) combination
    print("\nFinding optimal configurations...")
    optimal_configs = []
    
    for (kernel, N), group in df.groupby(['kernel_name', 'N']):
        # Find row with minimum execution time
        best_idx = group['exec_time'].idxmin()
        best_row = group.loc[best_idx]
        optimal_configs.append(best_row)
    
    df_optimal = pd.DataFrame(optimal_configs)
    print(f"✓ Found {len(df_optimal)} optimal configurations")
    
    # Define feature columns
    numeric_features = [
        'N',
        'dimensionality',
        'compute_intensity',
        'has_shared_memory',
        'global_reads',
        'global_writes',
        'arithmetic_ops',
        'memory_ops'
    ]
    
    # Add kernel name encoding if requested
    kernel_encoder = None
    if use_kernel_names and 'kernel_name' in df_optimal.columns:
        kernel_encoder = LabelEncoder()
        df_optimal['kernel_encoded'] = kernel_encoder.fit_transform(df_optimal['kernel_name'])
        numeric_features.insert(0, 'kernel_encoded')
        print(f"✓ Encoded {len(kernel_encoder.classes_)} unique kernel names")
    
    # Extract features
    X = df_optimal[numeric_features].values
    
    # Target: [log2(block_x), log2(block_y), log2(total_threads)]
    # Use log2 because block sizes are powers of 2
    y = np.column_stack([
        np.log2(df_optimal['block_x'].values),
        np.log2(df_optimal['block_y'].values),
        np.log2(df_optimal['block_x'].values * df_optimal['block_y'].values)
    ])
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target matrix: {y.shape}")
    print(f"Features: {numeric_features}")
    
    return X, y, numeric_features, kernel_encoder, df_optimal


def evaluate_model(trainer, test_loader, feature_names):
    """
    Comprehensive model evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Get predictions
    all_predictions = []
    all_actuals = []
    
    trainer.model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(trainer.device)
            outputs = trainer.model(batch_X).cpu().numpy()
            all_predictions.append(outputs)
            all_actuals.append(batch_y.numpy())
    
    predictions = np.vstack(all_predictions)
    actuals = np.vstack(all_actuals)
    
    # Convert from log2 to actual dimensions
    pred_dims = 2 ** predictions
    actual_dims = 2 ** actuals
    
    # Round to nearest power of 2
    pred_dims_rounded = 2 ** np.round(predictions)
    
    # Calculate metrics
    metrics = {}
    
    # 1. Exact match (all 3 values correct)
    exact_match = np.all(np.abs(predictions - actuals) < 0.5, axis=1).mean()
    metrics['exact_match_accuracy'] = exact_match * 100
    
    # 2. Close match (within 1 power of 2)
    close_match = np.all(np.abs(predictions - actuals) < 1.0, axis=1).mean()
    metrics['close_match_accuracy'] = close_match * 100
    
    # 3. Per-dimension accuracy
    for i, name in enumerate(['block_x', 'block_y', 'total_threads']):
        exact = np.abs(predictions[:, i] - actuals[:, i]) < 0.5
        close = np.abs(predictions[:, i] - actuals[:, i]) < 1.0
        metrics[f'{name}_exact'] = exact.mean() * 100
        metrics[f'{name}_close'] = close.mean() * 100
    
    # 4. MAE and R² in log space
    for i, name in enumerate(['block_x', 'block_y', 'total_threads']):
        mae = mean_absolute_error(actuals[:, i], predictions[:, i])
        r2 = r2_score(actuals[:, i], predictions[:, i])
        metrics[f'{name}_mae_log'] = mae
        metrics[f'{name}_r2_log'] = r2
    
    # Print results
    print(f"\nOverall Accuracy:")
    print(f"  Exact Match:  {metrics['exact_match_accuracy']:.2f}%")
    print(f"  Close Match:  {metrics['close_match_accuracy']:.2f}%")
    
    print(f"\nPer-Dimension Accuracy:")
    for dim in ['block_x', 'block_y', 'total_threads']:
        print(f"  {dim:15s}: Exact={metrics[f'{dim}_exact']:.2f}%, "
              f"Close={metrics[f'{dim}_close']:.2f}%, "
              f"R²={metrics[f'{dim}_r2_log']:.3f}")
    
    # Show example predictions
    print(f"\nSample Predictions (showing first 15):")
    print("-" * 70)
    print(f"{'Predicted':<30} {'Actual':<30} {'Match'}")
    print("-" * 70)
    
    for i in range(min(15, len(pred_dims_rounded))):
        bx_pred, by_pred, tot_pred = pred_dims_rounded[i]
        bx_act, by_act, tot_act = actual_dims[i]
        
        match = "✓" if np.all(np.abs(predictions[i] - actuals[i]) < 0.5) else "✗"
        
        print(f"({int(bx_pred):4d}, {int(by_pred):4d}, {int(tot_pred):4d})     "
              f"({int(bx_act):4d}, {int(by_act):4d}, {int(tot_act):4d})     {match}")
    
    return metrics


def plot_training_history(history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Training plot saved to: {save_path}")
    else:
        plt.show()


def predict_optimal_config(model_path, scaler_path, kernel_info):
    """
    Load trained model and predict optimal configuration
    
    Args:
        model_path: Path to saved model (.pth)
        scaler_path: Path to saved scaler (.pkl)
        kernel_info: Dict with kernel characteristics
        
    Returns:
        (block_x, block_y, total_threads)
    """
    # Load metadata
    with open(scaler_path, 'rb') as f:
        metadata = pickle.load(f)
    
    scaler = metadata['scaler']
    features = metadata['features']
    kernel_encoder = metadata.get('kernel_encoder')
    
    # Prepare input
    feature_values = []
    
    for feat in features:
        if feat == 'kernel_encoded' and kernel_encoder is not None:
            # Encode kernel name
            try:
                encoded = kernel_encoder.transform([kernel_info['kernel_name']])[0]
            except (KeyError, ValueError):
                # Unknown kernel, use mean encoding
                encoded = len(kernel_encoder.classes_) // 2
            feature_values.append(encoded)
        else:
            feature_values.append(kernel_info[feat])
    
    X = np.array([feature_values])
    X_scaled = scaler.transform(X)
    
    # Load model
    input_dim = len(features)
    model = BlockPredictor(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predict
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        output = model(X_tensor).numpy()[0]
    
    # Convert from log2 and round to powers of 2
    block_x = int(2 ** round(output[0]))
    block_y = int(2 ** round(output[1]))
    total_threads = int(2 ** round(output[2]))
    
    # Apply CUDA constraints
    block_x = max(32, min(block_x, 1024))
    block_y = max(1, min(block_y, 1024))
    
    # Ensure total threads <= 1024
    if block_x * block_y > 1024:
        # Adjust while maintaining aspect ratio
        ratio = block_x / block_y
        block_x = int(np.sqrt(1024 * ratio))
        block_y = 1024 // block_x
        # Round to powers of 2
        block_x = 2 ** int(np.log2(block_x))
        block_y = 2 ** int(np.log2(block_y))
    
    total_threads = block_x * block_y
    
    return block_x, block_y, total_threads


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Neural Network for CUDA Block Size Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          # Basic training
          python3 dnn_training.py klaraptor_enriched_data.csv
          
          # With custom parameters
          python3 dnn_training.py data.csv --epochs 300 --batch-size 64
          
          # Predict for new kernel
          python3 dnn_training.py --predict \\
              --kernel matrixMul --N 4096 --dim 2 --compute 15.2
        """
    )
    
    parser.add_argument('csv_file', nargs='?', help='Path to enriched CSV file')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128, 64, 32],
                       help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--no-kernel-names', action='store_true',
                       help='Do not use kernel names as features')
    
    # Prediction mode
    parser.add_argument('--predict', action='store_true', help='Prediction mode')
    parser.add_argument('--model', default='output/block_predictor_nn.pth')
    parser.add_argument('--scaler', default='output/block_predictor_scaler.pkl')
    parser.add_argument('--kernel', help='Kernel name')
    parser.add_argument('--N', type=int, help='Data size')
    parser.add_argument('--dim', type=int, help='Dimensionality')
    parser.add_argument('--compute', type=float, help='Compute intensity')
    
    args = parser.parse_args()
    
    # Prediction mode
    if args.predict:
        if not all([args.kernel, args.N, args.dim, args.compute]):
            parser.error("Prediction mode requires: --kernel, --N, --dim, --compute")
        
        kernel_info = {
            'kernel_name': args.kernel,
            'N': args.N,
            'dimensionality': args.dim,
            'compute_intensity': args.compute,
            'has_shared_memory': 0,
            'global_reads': 10,
            'global_writes': 5,
            'arithmetic_ops': 100,
            'memory_ops': 15
        }
        
        bx, by, total = predict_optimal_config(args.model, args.scaler, kernel_info)
        print(f"\nPredicted optimal configuration:")
        print(f"  block_dims: ({bx}, {by}, 1)")
        print(f"  total_threads: {total}")
        return
    
    # Training mode
    if not args.csv_file:
        parser.error("CSV file required for training")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("NEURAL NETWORK BLOCK SIZE PREDICTOR")
    print("=" * 70)
    print(f"Data: {args.csv_file}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden layers: {args.hidden}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print()
    
    # Prepare data
    X, y, features, kernel_encoder, df_optimal = prepare_data(
        args.csv_file,
        use_kernel_names=not args.no_kernel_names
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = KernelDataset(X_train_scaled, y_train)
    test_dataset = KernelDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BlockPredictor(
        input_dim=len(features),
        hidden_dims=args.hidden,
        dropout=args.dropout
    )
    
    trainer = BlockPredictorTrainer(model, device=device)
    
    # Train
    history = trainer.train(
        train_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=25
    )
    
    # Evaluate
    metrics = evaluate_model(trainer, test_loader, features)
    
    # Save model
    trainer.save(
        str(output_dir / 'block_predictor_nn.pth'),
        str(output_dir / 'block_predictor_scaler.pkl'),
        {
            'scaler': scaler,
            'features': features,
            'kernel_encoder': kernel_encoder
        }
    )
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {output_dir/'metrics.json'}")
    
    # Plot history
    plot_training_history(history, save_path=str(output_dir / 'training_history.png'))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTo predict for a new kernel:")
    print(f"  python3 {__file__} --predict \\")
    print(f"    --kernel myKernel --N 4096 --dim 2 --compute 12.5")


if __name__ == "__main__":
    main()