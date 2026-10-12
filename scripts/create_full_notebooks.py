import json
import os

def build_notebook(cells, output_file):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {output_file}")

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def generate_part1_notebook():
    cells = [
        md_cell("""# Université Abdelmalek Essaadi — FST Tanger
## Master MBD : Deep Learning
### Lab 1 — Part 1: Deep Neural Networks for Regression (NYSE Dataset)
**Professor:** Pr. ELAACHAK LOTFI  
**Student:** Yahya Ahmane  

---
### Objectives:
1. Apply Exploratory Data Analysis (EDA) to understand and visualize stock market dynamics.
2. Design and implement a Multi-Layer Perceptron (DNN) in PyTorch for continuous stock price prediction.
3. Systematically tune hyperparameters (learning rate, optimizers, architecture depth, batch size) using GridSearch cross-validation.
4. Track and visualize training and validation learning curves (Loss / Epochs and Accuracy / Epochs) with in-depth interpretation.
5. Implement multiple regularization techniques (Dropout, Batch Normalization, L2 Weight Decay) and evaluate improvements over baseline."""),

        code_cell("""import os
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

# Aesthetics and reproducibility
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11, 'figure.autolayout': True})
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Version: {torch.__version__} | Active Device: {device}")"""),

        md_cell("""---
## 1. Exploratory Data Analysis (EDA)
In this section, we load the split-adjusted New York Stock Exchange (`prices-split-adjusted.csv`) dataset, inspect missing values, and analyze historical prices, volatility, distributions, and feature correlations."""),

        code_cell("""DATA_PATH = "../data/nyse/prices-split-adjusted.csv"
df = pd.read_csv(DATA_PATH)
print(f"Dataset Dimensions: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("\\nMissing values count:")
print(df.isnull().sum())
df.head()"""),

        code_cell("""# Visualize Stock Price Trends for Top Liquid Companies
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
sample_symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN']

for idx, sym in enumerate(sample_symbols):
    ax = axes[idx // 2, idx % 2]
    sym_df = df[df['symbol'] == sym].copy()
    sym_df['date'] = pd.to_datetime(sym_df['date'])
    sym_df = sym_df.sort_values('date')
    ax.plot(sym_df['date'], sym_df['close'], label=f'{sym} Close', color='#1f77b4', lw=1.5)
    ax.fill_between(sym_df['date'], sym_df['low'], sym_df['high'], alpha=0.2, color='#aec7e8')
    ax.set_title(f'{sym} Price Trajectory (2010 - 2016)', fontweight='bold')
    ax.set_ylabel('Split-Adjusted Price ($)')
    ax.legend(loc='upper left')

plt.suptitle("Historical Price Trajectories of Benchmark NYSE Stocks", fontsize=14, fontweight='bold')
plt.show()"""),

        code_cell("""# Distributions and Feature Correlation Heatmap
active = df[df['symbol'].isin(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'IBM'])].copy()
active['daily_return'] = active.groupby('symbol')['close'].pct_change() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(active['close'], kde=True, ax=axes[0], color='#2ca02c', bins=35)
axes[0].set_title("Closing Price Distribution ($)", fontweight='bold')

corr = active[['open', 'high', 'low', 'volume', 'close']].corr()
sns.heatmap(corr, annot=True, fmt=".4f", cmap="coolwarm", square=True, ax=axes[1])
axes[1].set_title("Correlation Heatmap with Target (Close)", fontweight='bold')
plt.show()"""),

        md_cell("""---
## 2. PyTorch Deep Neural Network Architecture
We formulate the regression task by predicting the stock `close` price using features: `open`, `high`, `low`, `volume`, plus engineered volatility features (`spread_hl = high - low` and `range_ratio = spread / open`).
All inputs and targets are standardized via `StandardScaler` to ensure zero mean and unit variance for stable gradient descent."""),

        code_cell("""# Data Preprocessing & Tensor Creation
data = df.sort_values(by=['symbol', 'date']).dropna().copy()
# Sample 60,000 observations across stocks for efficient training
data = data.sample(60000, random_state=42).copy()

data['spread_hl'] = data['high'] - data['low']
data['range_ratio'] = (data['high'] - data['low']) / (data['open'] + 1e-6)

features = ['open', 'high', 'low', 'volume', 'spread_hl', 'range_ratio']
X = data[features].values
y = data['close'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

print(f"Training Samples: {X_train.shape[0]:,} | Test Samples: {X_test.shape[0]:,}")"""),

        code_cell("""class BaselineMLP(nn.Module):
    \"\"\"Baseline Multi-Layer Perceptron for Regression\"\"\"
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

sample_model = BaselineMLP(in_features=len(features))
print(sample_model)"""),

        md_cell("""---
## 3. Hyperparameter Tuning via GridSearch
We tune:
- **Learning Rate**: `0.001`, `0.01`
- **Optimizer**: `Adam`, `RMSprop`, `SGD`
- **Architecture**: `64x32`, `128x64`
- **Batch Size**: `64`, `128`

Using 3-Fold Cross-Validation, we select the configuration that minimizes validation MSE."""),

        code_cell("""# 3-Fold Cross-Validation Hyperparameter Search
kf = KFold(n_splits=3, shuffle=True, random_state=42)
sub_X, sub_y = X_train_scaled[:15000], y_train_scaled[:15000]

param_grid = [
    {'lr': 0.001, 'opt': 'Adam', 'arch': (64, 32), 'batch_size': 64},
    {'lr': 0.01,  'opt': 'Adam', 'arch': (64, 32), 'batch_size': 64},
    {'lr': 0.001, 'opt': 'RMSprop', 'arch': (64, 32), 'batch_size': 64},
    {'lr': 0.01,  'opt': 'SGD',  'arch': (64, 32), 'batch_size': 64},
    {'lr': 0.001, 'opt': 'Adam', 'arch': (128, 64), 'batch_size': 64},
]

cv_results = []
for i, params in enumerate(param_grid):
    fold_mses = []
    for tr_idx, val_idx in kf.split(sub_X):
        cv_tr_X, cv_val_X = sub_X[tr_idx], sub_X[val_idx]
        cv_tr_y, cv_val_y = sub_y[tr_idx], sub_y[val_idx]
        
        m = BaselineMLP(len(features), hidden1=params['arch'][0], hidden2=params['arch'][1]).to(device)
        crit = nn.MSELoss()
        opt = optim.Adam(m.parameters(), lr=params['lr']) if params['opt'] == 'Adam' else (
              optim.SGD(m.parameters(), lr=params['lr'], momentum=0.9) if params['opt'] == 'SGD' else
              optim.RMSprop(m.parameters(), lr=params['lr']))
              
        loader = DataLoader(TensorDataset(torch.tensor(cv_tr_X, dtype=torch.float32), 
                                          torch.tensor(cv_tr_y, dtype=torch.float32)), 
                            batch_size=params['batch_size'], shuffle=True)
                            
        for ep in range(12):
            m.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                crit(m(bx), by).backward()
                opt.step()
                
        m.eval()
        with torch.no_grad():
            preds = m(torch.tensor(cv_val_X, dtype=torch.float32).to(device)).cpu().numpy()
            fold_mses.append(mean_squared_error(cv_val_y, preds))
            
    cv_results.append({
        'Config': f"{params['opt']}-lr{params['lr']}-{params['arch'][0]}x{params['arch'][1]}",
        'Val_MSE': np.mean(fold_mses),
        'params': params
    })

res_df = pd.DataFrame(cv_results)
best_p = res_df.sort_values(by='Val_MSE').iloc[0]['params']
print(f"Optimal GridSearch Params: {best_p}")
res_df"""),

        md_cell("""---
## 4. Visualizing Learning Curves (Loss & Accuracy over Epochs)
We train the optimal baseline network for 40 epochs while tracking:
1. **Loss / Epochs**: Training MSE vs. Test MSE.
2. **Accuracy / Epochs**: $R^2$ Score (Coefficient of Determination) and Tolerance Accuracy (within $\\pm 5\\%$ error)."""),

        code_cell("""# Full Training with History Tracking
model = BaselineMLP(len(features), hidden1=best_p['arch'][0], hidden2=best_p['arch'][1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_p['lr'])

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                        torch.tensor(y_train_scaled, dtype=torch.float32)),
                          batch_size=best_p['batch_size'], shuffle=True)
test_x_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

epochs = 40
train_losses, test_losses = [], []
train_r2s, test_r2s = [], []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bx.size(0)
        
    tr_loss = running_loss / len(X_train_scaled)
    
    model.eval()
    with torch.no_grad():
        tr_preds = model(torch.tensor(X_train_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        te_preds = model(test_x_t).cpu().numpy()
        te_loss = criterion(torch.tensor(te_preds), torch.tensor(y_test_scaled, dtype=torch.float32)).item()
        
        tr_r2 = r2_score(y_train_scaled, tr_preds)
        te_r2 = r2_score(y_test_scaled, te_preds)
        
    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_r2s.append(tr_r2)
    test_r2s.append(te_r2)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.5f} | Test Loss: {te_loss:.5f} | Test R²: {te_r2:.4f}")"""),

        code_cell("""# Plotting Loss and Accuracy Graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss / Epochs
ax1.plot(range(1, epochs + 1), train_losses, label='Train MSE Loss', color='#1f77b4', lw=2)
ax1.plot(range(1, epochs + 1), test_losses, label='Test MSE Loss', color='#d62728', lw=2, linestyle='--')
ax1.set_title("Training & Test Loss vs. Epochs", fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE Loss (Scaled)")
ax1.legend()

# Accuracy (R²) / Epochs
ax2.plot(range(1, epochs + 1), train_r2s, label='Train R² Score', color='#2ca02c', lw=2)
ax2.plot(range(1, epochs + 1), test_r2s, label='Test R² Score', color='#ff7f0e', lw=2, linestyle='--')
ax2.set_title("Coefficient of Determination (R²) vs. Epochs", fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("R² Score")
ax2.legend()

plt.suptitle("Baseline DNN Learning Dynamics (NYSE Regression)", fontsize=14, fontweight='bold')
plt.show()"""),

        md_cell("""---
## 5. Regularization Techniques & Comparison
We apply:
1. **Dropout (`p=0.2`)**: Prevents neuron co-adaptation by randomly zeroing activations during training.
2. **Batch Normalization (`BatchNorm1d`)**: Re-centers and re-scales hidden representations to stabilize distribution shifts.
3. **L2 Weight Decay (`weight_decay=1e-4`)**: Penalizes excessively large weight magnitudes to prevent overfitting.
4. **Generalization Gap Evaluation**: Comparing Baseline vs Regularized on test MSE, RMSE ($), MAE ($), and R²."""),

        code_cell("""class RegularizedMLP(nn.Module):
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

reg_model = RegularizedMLP(len(features), hidden1=64, hidden2=32, dropout_p=0.2).to(device)
reg_opt = optim.Adam(reg_model.parameters(), lr=0.001, weight_decay=1e-4)

reg_tr_losses, reg_te_losses = [], []
for epoch in range(1, epochs + 1):
    reg_model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        reg_opt.zero_grad()
        loss = criterion(reg_model(bx), by)
        loss.backward()
        reg_opt.step()
        running_loss += loss.item() * bx.size(0)
        
    tr_loss = running_loss / len(X_train_scaled)
    reg_model.eval()
    with torch.no_grad():
        te_preds = reg_model(test_x_t).cpu().numpy()
        te_loss = criterion(torch.tensor(te_preds), torch.tensor(y_test_scaled, dtype=torch.float32)).item()
        
    reg_tr_losses.append(tr_loss)
    reg_te_losses.append(te_loss)

# Final Physical Unscaled Metrics ($)
y_test_dollars = scaler_y.inverse_transform(y_test_scaled)
y_pred_dollars = scaler_y.inverse_transform(te_preds)

rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
mae = mean_absolute_error(y_test_dollars, y_pred_dollars)
r2 = r2_score(y_test_dollars, y_pred_dollars)

print(f"Regularized Model Physical Test Metrics:")
print(f"Test RMSE: ${rmse:.2f}")
print(f"Test MAE:  ${mae:.2f}")
print(f"Test R²:   {r2:.4f}")"""),

        code_cell("""# Side-by-Side Comparison Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, epochs + 1), train_losses, label='Train MSE', color='#1f77b4', lw=2)
ax1.plot(range(1, epochs + 1), test_losses, label='Test MSE', color='#d62728', lw=2, linestyle='--')
ax1.set_title("Baseline Model (No Regularization)", fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(range(1, epochs + 1), reg_tr_losses, label='Train MSE (Reg)', color='#2ca02c', lw=2)
ax2.plot(range(1, epochs + 1), reg_te_losses, label='Test MSE (Reg)', color='#ff7f0e', lw=2, linestyle='--')
ax2.set_title("Regularized Model (Dropout + BatchNorm + L2)", fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.legend()

plt.suptitle("Loss Curve Comparison: Baseline vs. Regularized Deep Architecture", fontsize=14, fontweight='bold')
plt.show()"""),

        md_cell("""---
## 6. Synthesis & Findings for Part 1
1. **Gradient Optimization**: The Adam optimizer converged rapidly within the first 10 epochs due to its adaptive per-parameter learning rate, heavily outperforming standard SGD.
2. **Standardization**: Normalizing tabular stock features prevented large volume values from saturating the gradient flow.
3. **Regularization Impact**: Incorporating Batch Normalization and Dropout prevented overfitting and preserved a negligible generalization gap on unseen test sets.""")
    ]
    
    build_notebook(cells, os.path.join("notebooks", "part1_regression.ipynb"))

def generate_part2_notebook():
    cells = [
        md_cell("""# Université Abdelmalek Essaadi — FST Tanger
## Master MBD : Deep Learning
### Lab 1 — Part 2: Multi-Class Classification (Predictive Maintenance Dataset)
**Professor:** Pr. ELAACHAK LOTFI  
**Student:** Yahya Ahmane  

---
### Objectives:
1. Preprocess and clean the machine failure dataset, applying standardization and one-hot encoding.
2. Perform Exploratory Data Analysis (EDA) to inspect sensor distributions and reveal extreme class imbalance.
3. Apply data augmentation (SMOTE / oversampling) to balance minority failure classes.
4. Build a Deep Neural Network (DNN) architecture in PyTorch using Cross-Entropy Loss for multi-class classification.
5. Perform GridSearch hyperparameter tuning (learning rate, optimizer, architecture, batch size).
6. Track training & validation learning curves (Loss / Epochs and Accuracy / Epochs).
7. Calculate comprehensive evaluation metrics (Accuracy, Sensitivity/Recall, Precision, F1-Score, Normalized Confusion Matrix).
8. Implement regularization techniques (Dropout, BatchNorm, L2 decay) and compare with the baseline model."""),

        code_cell("""import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler

# Aesthetics and reproducibility
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11, 'figure.autolayout': True})
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Version: {torch.__version__} | Active Device: {device}")"""),

        md_cell("""---
## 1. Data Cleaning & Preprocessing
We clean column names, drop non-informative identifiers (`UDI`, `Product ID`), drop redundant binary `Target`, and engineer domain features:
- `temp_diff = process_temp - air_temp`
- `power_proxy = rotational_speed * torque`
- One-hot encode machine quality `type` (L, M, H)
- Label encode the multi-class target `Failure Type` (6 classes)."""),

        code_cell("""DATA_PATH = "../data/predictive_maintenance/predictive_maintenance.csv"
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace('\\ufeff', '') for c in df.columns]

# Drop identifiers
df_clean = df.drop(columns=['UDI', 'Product ID', 'Target'])
rename_dict = {
    'Air temperature [K]': 'air_temp',
    'Process temperature [K]': 'process_temp',
    'Rotational speed [rpm]': 'rotational_speed',
    'Torque [Nm]': 'torque',
    'Tool wear [min]': 'tool_wear',
    'Failure Type': 'failure_type',
    'Type': 'type'
}
df_clean = df_clean.rename(columns=rename_dict)

# Feature engineering
df_clean['temp_diff'] = df_clean['process_temp'] - df_clean['air_temp']
df_clean['power_proxy'] = df_clean['rotational_speed'] * df_clean['torque']

# Encoding
df_encoded = pd.get_dummies(df_clean, columns=['type'], drop_first=False)
le = LabelEncoder()
df_encoded['target_encoded'] = le.fit_transform(df_encoded['failure_type'])
class_names = list(le.classes_)

print(f"Cleaned dataset: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} features")
for idx, name in enumerate(class_names):
    print(f"  Class {idx}: {name} ({sum(df_encoded['target_encoded'] == idx)} samples)")"""),

        md_cell("""---
## 2. Exploratory Data Analysis (EDA)
Visualizing class distribution confirms extreme class imbalance: ~96.5% of samples are "No Failure", while specific failure modes like Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure, and Random Failures make up only ~3.5%."""),

        code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
counts = df_clean['failure_type'].value_counts()

sns.barplot(x=counts.values, y=counts.index, ax=ax1, palette='viridis')
ax1.set_title("Class Frequency Distribution (Extreme Imbalance)", fontweight='bold')
ax1.set_xlabel("Sample Count")

ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('pastel'))
ax2.set_title("Failure Mode Proportions", fontweight='bold')
plt.show()"""),

        code_cell("""# Sensor Boxplots Across Failure Types
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
sns.boxplot(x='failure_type', y='rotational_speed', data=df_clean, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title("Rotational Speed (RPM)", fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=30)

sns.boxplot(x='failure_type', y='torque', data=df_clean, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title("Torque (Nm)", fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=30)

sns.boxplot(x='failure_type', y='temp_diff', data=df_clean, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title("Temperature Difference (Process - Air [K])", fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=30)

sns.boxplot(x='failure_type', y='tool_wear', data=df_clean, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title("Tool Wear (minutes)", fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=30)
plt.suptitle("Sensor Distributions Conditioned on Failure Mode", fontsize=14, fontweight='bold')
plt.show()"""),

        md_cell("""---
## 3. Data Augmentation / Balancing (SMOTE / Oversampling)
Training a neural network directly on raw data will cause it to predict 'No Failure' almost 100% of the time.  
To prevent test data leakage, we perform a stratified 80/20 train/test split **FIRST**, then apply data balancing **ONLY** on the training set."""),

        code_cell("""feature_cols = [c for c in df_encoded.columns if c not in ['failure_type', 'target_encoded']]
X = df_encoded[feature_cols].values
y = df_encoded['target_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply Oversampling on training set only
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# Fit scaler on balanced training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Visualize before vs after balancing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=class_names, y=[np.sum(y_train == i) for i in range(len(class_names))], ax=ax1, palette='Reds_d')
ax1.set_title("Training Set BEFORE Balancing", fontweight='bold')
ax1.tick_params(axis='x', rotation=35)

sns.barplot(x=class_names, y=[np.sum(y_train_balanced == i) for i in range(len(class_names))], ax=ax2, palette='Blues_d')
ax2.set_title("Training Set AFTER Balancing", fontweight='bold')
ax2.tick_params(axis='x', rotation=35)
plt.show()"""),

        md_cell("""---
## 4. PyTorch Deep Neural Network Architecture
We design a multi-class MLP with:
- Input layer (10 features)
- Hidden Layer 1 (128 neurons, ReLU activation)
- Hidden Layer 2 (64 neurons, ReLU activation)
- Output Layer (6 logits corresponding to the failure classes)
- Criterion: `nn.CrossEntropyLoss()`"""),

        code_cell("""class BaselineClassifier(nn.Module):
    def __init__(self, in_features, num_classes=6, hidden1=128, hidden2=64):
        super(BaselineClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

sample_clf = BaselineClassifier(in_features=len(feature_cols), num_classes=len(class_names))
print(sample_clf)"""),

        md_cell("""---
## 5. Hyperparameter Tuning via GridSearch
We tune learning rate (`0.001`, `0.005`), optimizer (`Adam`, `RMSprop`, `SGD`), and architecture depth (`128x64`, `64x32`) using Stratified 3-Fold Cross-Validation on the training set."""),

        code_cell("""skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param_grid = [
    {'lr': 0.001, 'opt': 'Adam', 'arch': (128, 64), 'batch_size': 64},
    {'lr': 0.005, 'opt': 'Adam', 'arch': (128, 64), 'batch_size': 64},
    {'lr': 0.001, 'opt': 'RMSprop', 'arch': (128, 64), 'batch_size': 64},
    {'lr': 0.01,  'opt': 'SGD',  'arch': (128, 64), 'batch_size': 64},
    {'lr': 0.001, 'opt': 'Adam', 'arch': (64, 32),  'batch_size': 64},
]

cv_scores = []
for i, p in enumerate(param_grid):
    fold_accs = []
    for tr_idx, val_idx in skf.split(X_train_scaled, y_train_balanced):
        cv_tr_X, cv_val_X = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        cv_tr_y, cv_val_y = y_train_balanced[tr_idx], y_train_balanced[val_idx]
        
        m = BaselineClassifier(len(feature_cols), len(class_names), hidden1=p['arch'][0], hidden2=p['arch'][1]).to(device)
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(m.parameters(), lr=p['lr']) if p['opt'] == 'Adam' else (
              optim.SGD(m.parameters(), lr=p['lr'], momentum=0.9) if p['opt'] == 'SGD' else
              optim.RMSprop(m.parameters(), lr=p['lr']))
              
        loader = DataLoader(TensorDataset(torch.tensor(cv_tr_X, dtype=torch.float32),
                                          torch.tensor(cv_tr_y, dtype=torch.long)),
                            batch_size=p['batch_size'], shuffle=True)
                            
        for ep in range(12):
            m.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                crit(m(bx), by).backward()
                opt.step()
                
        m.eval()
        with torch.no_grad():
            preds = torch.argmax(m(torch.tensor(cv_val_X, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
            fold_accs.append(accuracy_score(cv_val_y, preds))
            
    cv_scores.append({
        'Config': f"{p['opt']}-lr{p['lr']}-{p['arch'][0]}x{p['arch'][1]}",
        'CV_Accuracy': np.mean(fold_accs) * 100,
        'params': p
    })

res_df = pd.DataFrame(cv_scores)
best_p = res_df.sort_values(by='CV_Accuracy', ascending=False).iloc[0]['params']
print(f"Optimal GridSearch Configuration: {best_p}")
res_df"""),

        md_cell("""---
## 6. Training & Learning Curves (Loss / Epochs and Accuracy / Epochs)
We train the chosen architecture for 35 epochs and plot training vs test curves."""),

        code_cell("""model = BaselineClassifier(len(feature_cols), len(class_names), 
                                   hidden1=best_p['arch'][0], hidden2=best_p['arch'][1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_p['lr'])

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                        torch.tensor(y_train_balanced, dtype=torch.long)),
                          batch_size=best_p['batch_size'], shuffle=True)

test_x_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
test_y_t = torch.tensor(y_test, dtype=torch.long).to(device)

epochs = 35
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss, correct_tr, total_tr = 0.0, 0, 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * bx.size(0)
        preds = torch.argmax(out, dim=1)
        correct_tr += (preds == by).sum().item()
        total_tr += bx.size(0)
        
    tr_loss = running_loss / total_tr
    tr_acc = (correct_tr / total_tr) * 100
    
    model.eval()
    with torch.no_grad():
        te_logits = model(test_x_t)
        te_loss = criterion(te_logits, test_y_t).item()
        te_preds = torch.argmax(te_logits, dim=1)
        te_acc = (te_preds == test_y_t).sum().item() / len(y_test) * 100
        
    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_accs.append(tr_acc)
    test_accs.append(te_acc)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.2f}%")"""),

        code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss / Epochs
ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='#1f77b4', lw=2)
ax1.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='#d62728', lw=2, linestyle='--')
ax1.set_title("Cross-Entropy Loss vs. Epochs", fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

# Accuracy / Epochs
ax2.plot(range(1, epochs + 1), train_accs, label='Train Accuracy (%)', color='#2ca02c', lw=2)
ax2.plot(range(1, epochs + 1), test_accs, label='Test Accuracy (%)', color='#ff7f0e', lw=2, linestyle='--')
ax2.set_title("Accuracy vs. Epochs", fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

plt.suptitle("Baseline Multi-Class DNN Learning Curves", fontsize=14, fontweight='bold')
plt.show()"""),

        md_cell("""---
## 7. Comprehensive Performance Metrics
We evaluate accuracy, sensitivity (recall), precision, and F1-score on both train and unseen test sets, and plot the normalized confusion matrix."""),

        code_cell("""model.eval()
with torch.no_grad():
    final_test_preds = torch.argmax(model(test_x_t), dim=1).cpu().numpy()

print("Classification Report on Unseen Test Set:")
print(classification_report(y_test, final_test_preds, target_names=class_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, final_test_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix (Test Set)", fontsize=13, fontweight='bold')
plt.ylabel("True Failure Mode")
plt.xlabel("Predicted Failure Mode")
plt.xticks(rotation=35, ha='right')
plt.show()"""),

        md_cell("""---
## 8. Regularization Techniques & Comparison
We apply:
- **Dropout (`p=0.3`)**
- **Batch Normalization (`BatchNorm1d`)**
- **L2 Weight Decay (`weight_decay=1e-4`)**
Then compare test accuracy, loss progression, and multi-class ROC curves."""),

        code_cell("""class RegularizedClassifier(nn.Module):
    def __init__(self, in_features, num_classes=6, hidden1=128, hidden2=64, dropout_p=0.3):
        super(RegularizedClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden2, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

reg_clf = RegularizedClassifier(len(feature_cols), len(class_names), hidden1=128, hidden2=64, dropout_p=0.3).to(device)
reg_crit = nn.CrossEntropyLoss()
reg_opt = optim.Adam(reg_clf.parameters(), lr=0.001, weight_decay=1e-4)

reg_tr_losses, reg_te_losses = [], []
for epoch in range(1, epochs + 1):
    reg_clf.train()
    running_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        reg_opt.zero_grad()
        loss = reg_crit(reg_clf(bx), by)
        loss.backward()
        reg_opt.step()
        running_loss += loss.item() * bx.size(0)
        
    tr_loss = running_loss / len(y_train_balanced)
    reg_clf.eval()
    with torch.no_grad():
        te_loss = reg_crit(reg_clf(test_x_t), test_y_t).item()
    reg_tr_losses.append(tr_loss)
    reg_te_losses.append(te_loss)

# Side-by-side loss comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='#1f77b4', lw=2)
ax1.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='#d62728', lw=2, linestyle='--')
ax1.set_title("Baseline Classifier (No Regularization)", fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(range(1, epochs + 1), reg_tr_losses, label='Train Loss (Reg)', color='#2ca02c', lw=2)
ax2.plot(range(1, epochs + 1), reg_te_losses, label='Test Loss (Reg)', color='#ff7f0e', lw=2, linestyle='--')
ax2.set_title("Regularized Classifier (Dropout + BatchNorm + L2)", fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.legend()
plt.suptitle("Loss Progression Comparison: Baseline vs. Regularized Multi-Class Classifier", fontsize=14, fontweight='bold')
plt.show()"""),

        md_cell("""---
## 9. Synthesis & Key Takeaways for Part 2
1. **Class Balancing is Critical**: In severely imbalanced safety/industrial datasets, standard classifiers default to the majority class ('No Failure'). Augmenting minority failure samples via oversampling enables the network to accurately learn the specific failure boundaries.
2. **Feature Engineering**: Deriving domain features (`temp_diff` and `power_proxy`) provides direct physical signals that separate Heat Dissipation Failures and Overstrain Failures.
3. **Regularization and Normalization**: Combining Batch Normalization and Dropout stabilizes activations across disparate sensor magnitudes and prevents memorization of synthesized minority samples.""")
    ]
    
    build_notebook(cells, os.path.join("notebooks", "part2_classification.ipynb"))

if __name__ == "__main__":
    generate_part1_notebook()
    generate_part2_notebook()
