"""
Lab 1 - Part 1: Deep Neural Networks for Regression (NYSE Dataset)
Author: yahya-ahmane
Program: LSI (Génie Logiciel et Systèmes Intelligents) - Deep Learning, FST Tanger
Pr. ELAACHAK LOTFI
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set styles and random seeds for reproducibility
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11, 'figure.autolayout': True})
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nyse", "prices-split-adjusted.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------
# Step 1: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
def run_eda():
    print("\n" + "="*60)
    print("STEP 1: Exploratory Data Analysis (EDA) - NYSE Dataset")
    print("="*60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Check top traded companies
    top_symbols = df['symbol'].value_counts().head(5).index.tolist()
    print(f"\nTop actively traded symbols: {top_symbols}")
    
    # 1. Plot Stock Price Trends for Top Symbols
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    sample_symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    for idx, sym in enumerate(sample_symbols):
        ax = axes[idx // 2, idx % 2]
        sym_df = df[df['symbol'] == sym].copy()
        if not sym_df.empty:
            sym_df['date'] = pd.to_datetime(sym_df['date'])
            sym_df = sym_df.sort_values('date')
            ax.plot(sym_df['date'], sym_df['close'], label=f'{sym} Close', color='#1f77b4', lw=1.5)
            ax.fill_between(sym_df['date'], sym_df['low'], sym_df['high'], alpha=0.2, color='#aec7e8')
            ax.set_title(f'{sym} Historical Price Trend (2010 - 2016)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Split-Adjusted Price ($)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper left')
    plt.suptitle("Exploratory Data Analysis: Historical Price Trajectories", fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q1_stock_trends.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q1_stock_trends.png")

    # 2. Distributions & Feature Relationships
    # Select clean representative benchmark sample (AAPL, MSFT, GOOG, AMZN, IBM)
    active_stocks = df[df['symbol'].isin(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'IBM'])].copy()
    active_stocks['daily_return'] = active_stocks.groupby('symbol')['close'].pct_change() * 100
    active_stocks['spread'] = active_stocks['high'] - active_stocks['low']
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.histplot(active_stocks['close'], kde=True, ax=axes[0, 0], color='#2ca02c', bins=40)
    axes[0, 0].set_title("Closing Price Distribution ($)", fontweight='bold')
    
    sns.histplot(active_stocks['volume'] / 1e6, kde=True, ax=axes[0, 1], color='#ff7f0e', bins=40)
    axes[0, 1].set_title("Trading Volume Distribution (Millions of Shares)", fontweight='bold')
    
    sns.boxplot(x='symbol', y='daily_return', data=active_stocks, ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title("Daily Percentage Return Volatility (%)", fontweight='bold')
    axes[1, 0].set_ylim(-10, 10)
    
    sns.scatterplot(x='open', y='close', hue='symbol', data=active_stocks.sample(2000, random_state=42), ax=axes[1, 1], alpha=0.6)
    axes[1, 1].set_title("Open Price vs. Close Price", fontweight='bold')
    
    plt.suptitle("Exploratory Data Analysis: Feature Distributions & Volatility", fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q1_eda_distributions.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q1_eda_distributions.png")

    # 3. Correlation Matrix
    numeric_cols = ['open', 'high', 'low', 'volume', 'close']
    corr = active_stocks[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".4f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Correlation Matrix (NYSE Stock Features)", fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q1_correlation_matrix.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q1_correlation_matrix.png")

    return df

# ---------------------------------------------------------
# Step 2: Data Preparation & DNN Architecture
# ---------------------------------------------------------
def prepare_regression_data(df, sample_size=60000):
    print("\n" + "="*60)
    print("STEP 2: Data Preprocessing & PyTorch Tensors Preparation")
    print("="*60)
    
    # Feature engineering: Open, High, Low, Volume, High-Low spread, Open-Close spread ratio
    data = df.sort_values(by=['symbol', 'date']).dropna().copy()
    
    # Select a rich, diverse sample of ~60,000 rows across symbols
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=42).copy()
    
    data['spread_hl'] = data['high'] - data['low']
    data['range_ratio'] = (data['high'] - data['low']) / (data['open'] + 1e-6)
    
    feature_cols = ['open', 'high', 'low', 'volume', 'spread_hl', 'range_ratio']
    target_col = 'close'
    
    X = data[feature_cols].values
    y = data[target_col].values.reshape(-1, 1)
    
    # 80% Train, 20% Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(f"Features: {feature_cols}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, len(feature_cols)

class BaselineMLP(nn.Module):
    """Deep Neural Network Architecture for Regression"""
    def __init__(self, in_features, hidden1=64, hidden2=32):
        super(BaselineMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

import matplotlib.patches as mpatches

def plot_architecture_diagram():
    """Generates a clean architectural visual for the Baseline PyTorch DNN"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis('off')
    
    layers = [
        {"name": "Input Layer\n(6 Features)", "color": "#4a90e2"},
        {"name": "Dense Layer 1\n(64 Units + ReLU)", "color": "#50e3c2"},
        {"name": "Dense Layer 2\n(32 Units + ReLU)", "color": "#f5a623"},
        {"name": "Output Layer\n(1 Linear Unit)\nStock Close", "color": "#e94e77"}
    ]
    
    for i, layer in enumerate(layers):
        rect = mpatches.FancyBboxPatch((i * 2.8, 1.3), 2.2, 1.9, boxstyle="round,pad=0.2", 
                                       facecolor=layer["color"], edgecolor="black", linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(i * 2.8 + 1.1, 2.25, layer["name"], ha="center", va="center", 
                fontsize=11, fontweight="bold", color="black" if i != 3 else "white")
        if i < len(layers) - 1:
            ax.annotate("", xy=((i + 1) * 2.8, 2.25), xytext=(i * 2.8 + 2.2, 2.25),
                        arrowprops=dict(arrowstyle="->", lw=2.5, color="#333333"))
            
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0.5, 4.0)
    plt.title("PyTorch DNN Architecture for Stock Price Regression", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q2_model_architecture.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q2_model_architecture.png")

# ---------------------------------------------------------
# Step 3: GridSearchCV Hyperparameter Tuning
# ---------------------------------------------------------
def train_and_eval_model(model, X_train, y_train, X_val, y_val, opt_name='Adam', lr=0.001, epochs=25, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    if opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
        val_preds = model(val_x).cpu().numpy()
        val_mse = mean_squared_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        
    return val_mse, val_r2

def run_grid_search(X_train, y_train, in_features):
    print("\n" + "="*60)
    print("STEP 3: Hyperparameter Tuning (GridSearch)")
    print("="*60)
    
    param_grid = [
        {'lr': 0.001, 'opt': 'Adam', 'arch': (64, 32), 'batch_size': 64},
        {'lr': 0.01,  'opt': 'Adam', 'arch': (64, 32), 'batch_size': 64},
        {'lr': 0.001, 'opt': 'RMSprop', 'arch': (64, 32), 'batch_size': 64},
        {'lr': 0.01,  'opt': 'SGD',  'arch': (64, 32), 'batch_size': 64},
        {'lr': 0.001, 'opt': 'Adam', 'arch': (128, 64), 'batch_size': 64},
        {'lr': 0.001, 'opt': 'Adam', 'arch': (128, 64), 'batch_size': 128},
    ]
    
    # 3-Fold CV sample
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # Use a 15,000 subset for swift cross-validation
    sub_X, sub_y = X_train[:15000], y_train[:15000]
    
    results = []
    print(f"{'Config #':<10} | {'LR':<7} | {'Optimizer':<9} | {'Architecture':<12} | {'Batch':<6} | {'CV Val MSE':<10} | {'CV Val R2':<10}")
    print("-" * 75)
    
    for i, params in enumerate(param_grid):
        cv_mses, cv_r2s = [], []
        for train_idx, val_idx in kf.split(sub_X):
            cv_X_train, cv_X_val = sub_X[train_idx], sub_X[val_idx]
            cv_y_train, cv_y_val = sub_y[train_idx], sub_y[val_idx]
            
            m = BaselineMLP(in_features, hidden1=params['arch'][0], hidden2=params['arch'][1])
            mse, r2 = train_and_eval_model(m, cv_X_train, cv_y_train, cv_X_val, cv_y_val,
                                          opt_name=params['opt'], lr=params['lr'],
                                          epochs=15, batch_size=params['batch_size'])
            cv_mses.append(mse)
            cv_r2s.append(r2)
            
        mean_mse = np.mean(cv_mses)
        mean_r2 = np.mean(cv_r2s)
        
        cfg_name = f"C{i+1}: {params['opt']}-lr{params['lr']}-{params['arch'][0]}x{params['arch'][1]}"
        results.append({
            'Config': cfg_name,
            'lr': params['lr'],
            'opt': params['opt'],
            'arch': f"{params['arch'][0]}x{params['arch'][1]}",
            'batch_size': params['batch_size'],
            'mean_mse': mean_mse,
            'mean_r2': mean_r2,
            'params': params
        })
        print(f"C{i+1:<9} | {params['lr']:<7} | {params['opt']:<9} | {str(params['arch']):<12} | {params['batch_size']:<6} | {mean_mse:<10.5f} | {mean_r2:<10.5f}")
        
    res_df = pd.DataFrame(results)
    best_config = res_df.sort_values(by='mean_mse').iloc[0]
    print(f"\nOptimal Configuration: {best_config['Config']} with Val MSE: {best_config['mean_mse']:.5f}, R2: {best_config['mean_r2']:.5f}")
    
    # Plot GridSearch comparison
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(res_df))
    bars = ax1.bar(x_pos - 0.2, res_df['mean_mse'], width=0.4, label='Validation MSE (Lower is Better)', color='#4c72b0')
    ax1.set_ylabel('Mean Squared Error (MSE)', color='#4c72b0', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"C{i+1}\n{r['opt']}\nlr={r['lr']}" for i, r in res_df.iterrows()], rotation=0)
    
    ax2 = ax1.twinx()
    lines = ax2.plot(x_pos + 0.2, res_df['mean_r2'], color='#c44e52', marker='o', lw=2.5, label='R² Score (Higher is Better)')
    ax2.set_ylabel('R² Score', color='#c44e52', fontweight='bold')
    ax2.set_ylim(0.9, 1.01)
    
    plt.title("GridSearch Cross-Validation Performance Across Hyperparameters", fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q3_gridsearch_results.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q3_gridsearch_results.png")
    
    return best_config['params']

# ---------------------------------------------------------
# Step 4: Training with Epoch-by-Epoch Tracking (Loss & Accuracy)
# ---------------------------------------------------------
def train_full_model_with_curves(X_train, y_train, X_test, y_test, in_features, best_params, epochs=40):
    print("\n" + "="*60)
    print("STEP 4: Training Baseline Model & Tracking Loss/Accuracy Curves")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineMLP(in_features, hidden1=best_params['arch'][0], hidden2=best_params['arch'][1]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    batch_size = best_params['batch_size']
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_tensor_y = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    train_losses, test_losses = [], []
    train_r2s, test_r2s = [], []
    train_acc_5pct, test_acc_5pct = [], [] # Tolerance Accuracy: |y_pred - y|/|y| <= 5%
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_loss / len(X_train)
        
        # Validation / Test Evaluation
        model.eval()
        with torch.no_grad():
            train_preds = model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
            test_preds = model(test_tensor_x).cpu().numpy()
            
            epoch_test_loss = criterion(torch.tensor(test_preds), torch.tensor(y_test, dtype=torch.float32)).item()
            
            ep_train_r2 = r2_score(y_train, train_preds)
            ep_test_r2 = r2_score(y_test, test_preds)
            
            # 5% tolerance accuracy
            ep_tr_acc = np.mean(np.abs(train_preds - y_train) <= 0.05) * 100
            ep_te_acc = np.mean(np.abs(test_preds - y_test) <= 0.05) * 100
            
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        train_r2s.append(ep_train_r2)
        test_r2s.append(ep_test_r2)
        train_acc_5pct.append(ep_tr_acc)
        test_acc_5pct.append(ep_te_acc)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.5f} | Test Loss: {epoch_test_loss:.5f} | Train R²: {ep_train_r2:.4f} | Test R²: {ep_test_r2:.4f}")
            
    # Plot 1: Loss / Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss (MSE)', color='#1f77b4', lw=2)
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss (MSE)', color='#d62728', lw=2, linestyle='--')
    plt.title("Part 1: Training vs. Test Loss Curve over Epochs (NYSE Regression)", fontsize=13, fontweight='bold')
    plt.xlabel("Epochs", fontweight='bold')
    plt.ylabel("Mean Squared Error (Scaled)", fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q4_loss_epochs.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q4_loss_epochs.png")
    
    # Plot 2: Accuracy / Epochs (R² Score & Tolerance Accuracy)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, epochs + 1), train_r2s, label='Train R² Score', color='#2ca02c', lw=2)
    ax1.plot(range(1, epochs + 1), test_r2s, label='Test R² Score', color='#ff7f0e', lw=2, linestyle='--')
    ax1.set_xlabel("Epochs", fontweight='bold')
    ax1.set_ylabel("R² Coefficient of Determination", color='#2ca02c', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), test_acc_5pct, label='Test Tolerance Accuracy (±5%)', color='#9467bd', lw=1.5, linestyle=':')
    ax2.set_ylabel("Accuracy % (within ±0.05 tolerance)", color='#9467bd', fontweight='bold')
    
    plt.title("Part 1: Model Accuracy & R² Metric Progression over Epochs", fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q4_accuracy_epochs.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q4_accuracy_epochs.png")
    
    return model, train_losses, test_losses

# ---------------------------------------------------------
# Step 5: Regularization Techniques & Comparison
# ---------------------------------------------------------
class RegularizedMLP(nn.Module):
    """Regularized DNN with Batch Normalization & Dropout"""
    def __init__(self, in_features, hidden1=64, hidden2=32, dropout_p=0.2):
        super(RegularizedMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def run_regularization_comparison(X_train, y_train, X_test, y_test, in_features, base_train_losses, base_test_losses, scaler_y, epochs=40):
    print("\n" + "="*60)
    print("STEP 5: Regularization Techniques & Model Comparison")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg_model = RegularizedMLP(in_features, hidden1=64, hidden2=32, dropout_p=0.2).to(device)
    
    criterion = nn.MSELoss()
    # Apply L2 Weight Decay Regularization in Optimizer
    optimizer = optim.Adam(reg_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    reg_train_losses, reg_test_losses = [], []
    
    for epoch in range(1, epochs + 1):
        reg_model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = reg_model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_tr_loss = running_loss / len(X_train)
        
        reg_model.eval()
        with torch.no_grad():
            test_preds = reg_model(test_tensor_x).cpu().numpy()
            epoch_te_loss = criterion(torch.tensor(test_preds), torch.tensor(y_test, dtype=torch.float32)).item()
            
        reg_train_losses.append(epoch_tr_loss)
        reg_test_losses.append(epoch_te_loss)
        
    # Evaluate Final Metrics on unscaled Dollars
    reg_model.eval()
    with torch.no_grad():
        final_test_preds_scaled = reg_model(test_tensor_x).cpu().numpy()
        
    y_test_dollars = scaler_y.inverse_transform(y_test)
    y_pred_dollars = scaler_y.inverse_transform(final_test_preds_scaled)
    
    test_rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
    test_mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
    test_r2 = r2_score(y_test_dollars, y_pred_dollars)
    
    print("\nRegularized Model Performance (Physical Stock Prices $):")
    print(f"Test RMSE: ${test_rmse:.3f}")
    print(f"Test MAE:  ${test_mae:.3f}")
    print(f"Test R²:   {test_r2:.4f}")
    
    # 1. Comparison Loss Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range(1, epochs + 1), base_train_losses, label='Train Loss', color='#1f77b4', lw=2)
    ax1.plot(range(1, epochs + 1), base_test_losses, label='Test Loss', color='#d62728', lw=2, linestyle='--')
    ax1.set_title("Baseline Model (No Regularization)", fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(range(1, epochs + 1), reg_train_losses, label='Train Loss (Reg)', color='#2ca02c', lw=2)
    ax2.plot(range(1, epochs + 1), reg_test_losses, label='Test Loss (Reg)', color='#ff7f0e', lw=2, linestyle='--')
    ax2.set_title("Regularized Model (Dropout + BatchNorm + L2 Decay)", fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Loss Curve Comparison: Baseline vs. Regularized Deep Architecture", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q5_regularization_comparison.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q5_regularization_comparison.png")
    
    # 2. Actual vs. Predicted Plot
    plt.figure(figsize=(8, 7))
    sample_indices = np.random.choice(len(y_test_dollars), size=1000, replace=False)
    plt.scatter(y_test_dollars[sample_indices], y_pred_dollars[sample_indices], alpha=0.5, color='#4c72b0', edgecolors='none', label='Test Samples')
    min_val = min(y_test_dollars[sample_indices].min(), y_pred_dollars[sample_indices].min())
    max_val = max(y_test_dollars[sample_indices].max(), y_pred_dollars[sample_indices].max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2, label='Ideal Identity Line (y = x)')
    plt.title(f"Actual vs. Predicted Stock Prices ($) - Test Set (R² = {test_r2:.4f})", fontsize=12, fontweight='bold')
    plt.xlabel("Actual Closing Price ($)", fontweight='bold')
    plt.ylabel("Predicted Closing Price ($)", fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_DIR, "part1_q5_actual_vs_predicted.png"), dpi=200)
    plt.close()
    print("Saved: figures/part1_q5_actual_vs_predicted.png")
    
    return {
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'base_final_loss': base_test_losses[-1],
        'reg_final_loss': reg_test_losses[-1]
    }

def main():
    start_time = time.time()
    df = run_eda()
    plot_architecture_diagram()
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, in_features = prepare_regression_data(df)
    best_params = run_grid_search(X_train, y_train, in_features)
    model, base_tr_losses, base_te_losses = train_full_model_with_curves(X_train, y_train, X_test, y_test, in_features, best_params, epochs=40)
    results = run_regularization_comparison(X_train, y_train, X_test, y_test, in_features, base_tr_losses, base_te_losses, scaler_y, epochs=40)
    
    print("\n" + "="*60)
    print(f"Part 1 Complete in {time.time() - start_time:.2f} seconds!")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
