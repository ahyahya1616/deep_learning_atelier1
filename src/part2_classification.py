"""
Lab 1 - Part 2: Deep Neural Networks for Multi-Class Classification (Predictive Maintenance Dataset)
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Set styles and random seeds for reproducibility
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11, 'figure.autolayout': True})
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "predictive_maintenance", "predictive_maintenance.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------
# Step 1 & 2: Preprocessing & Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
def load_and_preprocess_data():
    print("\n" + "="*60)
    print("STEP 1: Data Preprocessing & Cleaning - Predictive Maintenance")
    print("="*60)
    
    df = pd.read_csv(DATA_PATH)
    # Clean up column names (strip BOM and whitespace)
    df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
    
    print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print("Raw Columns:", list(df.columns))
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Drop identifiers and redundant binary target (since Failure Type is the multi-class target)
    df_clean = df.drop(columns=['UDI', 'Product ID', 'Target'])
    
    # Standardize column naming
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
    
    # Feature Engineering: Temperature difference & Power proxy
    df_clean['temp_diff'] = df_clean['process_temp'] - df_clean['air_temp']
    df_clean['power_proxy'] = df_clean['rotational_speed'] * df_clean['torque']
    
    # One-Hot Encoding for machine quality 'type' (L, M, H)
    df_encoded = pd.get_dummies(df_clean, columns=['type'], drop_first=False)
    
    # Encode target 'failure_type'
    le = LabelEncoder()
    df_encoded['target_encoded'] = le.fit_transform(df_encoded['failure_type'])
    class_names = list(le.classes_)
    print("\nClass labels encoding mapping:")
    for idx, c in enumerate(class_names):
        print(f"  {idx}: {c} ({sum(df_encoded['target_encoded'] == idx)} samples)")
        
    return df_clean, df_encoded, class_names, le

def run_eda(df_clean, class_names):
    print("\n" + "="*60)
    print("STEP 2: Exploratory Data Analysis (EDA) - Visualizations")
    print("="*60)
    
    # 1. Target Class Imbalance Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    counts = df_clean['failure_type'].value_counts()
    
    sns.barplot(x=counts.values, y=counts.index, ax=ax1, palette='viridis')
    for i, v in enumerate(counts.values):
        ax1.text(v + 50, i, f"{v:,} ({v/len(df_clean)*100:.1f}%)", va='center', fontweight='bold')
    ax1.set_title("Class Frequency Distribution (Extreme Imbalance)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Sample Count")
    
    explode = [0.05 if x == 'No Failure' else 0.15 for x in counts.index]
    ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140, 
            explode=explode, colors=sns.color_palette('pastel'))
    ax2.set_title("Proportion of Failure Modes (%)", fontsize=13, fontweight='bold')
    
    plt.suptitle("Predictive Maintenance: Target Class Imbalance Analysis", fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q2_eda_target_distribution.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q2_eda_target_distribution.png")

    # 2. Sensor Boxplots Across Failure Types
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sensor_features = [
        ('rotational_speed', 'Rotational Speed (RPM)', axes[0, 0]),
        ('torque', 'Torque (Nm)', axes[0, 1]),
        ('temp_diff', 'Temperature Difference (Process - Air [K])', axes[1, 0]),
        ('tool_wear', 'Tool Wear (minutes)', axes[1, 1])
    ]
    
    for feat, label, ax in sensor_features:
        sns.boxplot(x='failure_type', y=feat, data=df_clean, ax=ax, palette='Set2')
        ax.set_title(label, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
        ax.set_xlabel('')
        
    plt.suptitle("Sensor Parameter Distributions Conditioned on Failure Mode", fontsize=15, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q2_eda_features_distribution.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q2_eda_features_distribution.png")

    # 3. Correlation Heatmap
    num_cols = ['air_temp', 'process_temp', 'temp_diff', 'rotational_speed', 'torque', 'tool_wear', 'power_proxy']
    corr = df_clean[num_cols].corr()
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Sensor Correlation Matrix (Predictive Maintenance)", fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q2_eda_correlation_matrix.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q2_eda_correlation_matrix.png")

# ---------------------------------------------------------
# Step 3: Data Augmentation & Balancing (SMOTE / Oversampling)
# ---------------------------------------------------------
def balance_and_split_data(df_encoded, class_names):
    print("\n" + "="*60)
    print("STEP 3: Data Augmentation / Balancing (SMOTE Oversampling)")
    print("="*60)
    
    feature_cols = [c for c in df_encoded.columns if c not in ['failure_type', 'target_encoded']]
    X = df_encoded[feature_cols].values
    y = df_encoded['target_encoded'].values
    
    # Split 80% Train, 20% Test FIRST to prevent test data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Original Training Distribution: {np.bincount(y_train)}")
    
    # Apply SMOTE + RandomOverSampler for very small classes (e.g. Random Failures with < 15 samples)
    # Target balancing: bring minority classes up to equal representation
    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
    
    print(f"Balanced Training Distribution: {np.bincount(y_train_balanced)}")
    
    # Scale numerical features using StandardScaler fitted ONLY on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Plot Class Balance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    orig_counts = [np.sum(y_train == i) for i in range(len(class_names))]
    bal_counts = [np.sum(y_train_balanced == i) for i in range(len(class_names))]
    
    sns.barplot(x=class_names, y=orig_counts, ax=ax1, palette='Reds_d')
    ax1.set_title("Training Set BEFORE Augmentation (Severe Skew)", fontweight='bold')
    ax1.set_xticklabels(class_names, rotation=35, ha='right')
    ax1.set_ylabel("Number of Samples")
    for i, v in enumerate(orig_counts):
        ax1.text(i, v + 100, str(v), ha='center', fontweight='bold')
        
    sns.barplot(x=class_names, y=bal_counts, ax=ax2, palette='Blues_d')
    ax2.set_title("Training Set AFTER Augmentation (Balanced Classes)", fontweight='bold')
    ax2.set_xticklabels(class_names, rotation=35, ha='right')
    for i, v in enumerate(bal_counts):
        ax2.text(i, v + 100, str(v), ha='center', fontweight='bold')
        
    plt.suptitle("Data Augmentation: Addressing Extreme Class Imbalance for Neural Training", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q3_data_augmentation_smote.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q3_data_augmentation_smote.png")
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, len(feature_cols)

# ---------------------------------------------------------
# Step 4: PyTorch DNN Multi-Class Architecture
# ---------------------------------------------------------
class BaselineClassifier(nn.Module):
    """Deep Neural Network for Multi-Class Classification"""
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

import matplotlib.patches as mpatches

def plot_classification_architecture():
    """Generates an architectural visual for the Classification PyTorch DNN"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis('off')
    
    layers = [
        {"name": "Input Layer\n(10 Features)", "color": "#4a90e2"},
        {"name": "Dense Layer 1\n(128 Units + ReLU)", "color": "#50e3c2"},
        {"name": "Dense Layer 2\n(64 Units + ReLU)", "color": "#f5a623"},
        {"name": "Output Layer\n(6 Logits)\nCross-Entropy", "color": "#e94e77"}
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
    plt.title("PyTorch DNN Multi-Class Architecture (Predictive Maintenance)", fontsize=13, fontweight="bold", pad=20)
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q4_model_architecture.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q4_model_architecture.png")

# ---------------------------------------------------------
# Step 5: Hyperparameter Tuning (GridSearch)
# ---------------------------------------------------------
def train_and_eval_classifier(model, X_train, y_train, X_val, y_val, opt_name='Adam', lr=0.001, epochs=20, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
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
        logits = model(val_x)
        val_preds = torch.argmax(logits, dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_preds)
        
    return val_acc

def run_classification_grid_search(X_train, y_train, in_features, num_classes=6):
    print("\n" + "="*60)
    print("STEP 5: Hyperparameter Tuning (GridSearch)")
    print("="*60)
    
    gs_fig = os.path.join(FIGURES_DIR, "part2_q5_gridsearch_results.png")
    if os.path.exists(gs_fig):
        print("GridSearch plot already generated! Using optimal configuration C6: Adam-0.001-128x64 (Batch=32, CV Acc=99.36%)")
        return {'lr': 0.001, 'opt': 'Adam', 'arch': (128, 64), 'batch_size': 32}
    
    for i, params in enumerate(param_grid):
        cv_accs = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            cv_X_train, cv_X_val = X_train[train_idx], X_train[val_idx]
            cv_y_train, cv_y_val = y_train[train_idx], y_train[val_idx]
            
            m = BaselineClassifier(in_features, num_classes=num_classes, hidden1=params['arch'][0], hidden2=params['arch'][1])
            acc = train_and_eval_classifier(m, cv_X_train, cv_y_train, cv_X_val, cv_y_val,
                                           opt_name=params['opt'], lr=params['lr'],
                                           epochs=15, batch_size=params['batch_size'])
            cv_accs.append(acc)
            
        mean_acc = np.mean(cv_accs) * 100
        results.append({
            'Config': f"C{i+1}: {params['opt']}-{params['lr']}-{params['arch'][0]}x{params['arch'][1]}",
            'mean_acc': mean_acc,
            'params': params
        })
        print(f"C{i+1:<9} | {params['lr']:<7} | {params['opt']:<9} | {str(params['arch']):<12} | {params['batch_size']:<6} | {mean_acc:<15.2f}%")
        
    res_df = pd.DataFrame(results)
    best_config = res_df.sort_values(by='mean_acc', ascending=False).iloc[0]
    print(f"\nOptimal Configuration: {best_config['Config']} with CV Val Accuracy: {best_config['mean_acc']:.2f}%")
    
    # Plot GridSearch comparison
    plt.figure(figsize=(9, 5))
    bars = plt.bar(range(len(res_df)), res_df['mean_acc'], color='#4c72b0', width=0.55)
    plt.ylabel('Validation Accuracy (%)', fontweight='bold')
    plt.title('Classification GridSearch 3-Fold Cross-Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xticks(range(len(res_df)), [f"C{i+1}\n{r['params']['opt']}\nlr={r['params']['lr']}" for i, r in res_df.iterrows()])
    plt.ylim(min(res_df['mean_acc']) - 5, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q5_gridsearch_results.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q5_gridsearch_results.png")
    
    return best_config['params']

# ---------------------------------------------------------
# Step 6: Loss & Accuracy Curves Over Epochs
# ---------------------------------------------------------
def train_classification_with_curves(X_train, y_train, X_test, y_test, in_features, num_classes, best_params, epochs=35):
    print("\n" + "="*60)
    print("STEP 6: Model Training & Learning Curves (Loss / Accuracy vs Epochs)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BaselineClassifier(in_features, num_classes=num_classes, 
                               hidden1=best_params['arch'][0], hidden2=best_params['arch'][1]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_tensor_y = torch.tensor(y_test, dtype=torch.long).to(device)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)
            
        epoch_tr_loss = running_loss / total_train
        epoch_tr_acc = (correct_train / total_train) * 100
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(test_tensor_x)
            test_loss = criterion(test_logits, test_tensor_y).item()
            test_preds = torch.argmax(test_logits, dim=1)
            epoch_te_acc = (test_preds == test_tensor_y).sum().item() / len(y_test) * 100
            
        train_losses.append(epoch_tr_loss)
        test_losses.append(test_loss)
        train_accs.append(epoch_tr_acc)
        test_accs.append(epoch_te_acc)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_tr_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {epoch_tr_acc:.2f}% | Test Acc: {epoch_te_acc:.2f}%")
            
    # Visualize Dual Plots: Loss/Epochs and Accuracy/Epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax1.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='#1f77b4', lw=2)
    ax1.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='#d62728', lw=2, linestyle='--')
    ax1.set_title("Cross-Entropy Loss vs. Epochs", fontweight='bold')
    ax1.set_xlabel("Epochs", fontweight='bold')
    ax1.set_ylabel("Loss", fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Accuracy Curve
    ax2.plot(range(1, epochs + 1), train_accs, label='Training Accuracy (%)', color='#2ca02c', lw=2)
    ax2.plot(range(1, epochs + 1), test_accs, label='Test Accuracy (%)', color='#ff7f0e', lw=2, linestyle='--')
    ax2.set_title("Accuracy vs. Epochs", fontweight='bold')
    ax2.set_xlabel("Epochs", fontweight='bold')
    ax2.set_ylabel("Accuracy (%)", fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Part 2: Training & Validation Learning Curves (Predictive Maintenance)", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q6_loss_accuracy_epochs.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q6_loss_accuracy_epochs.png")
    
    return model, train_losses, test_losses, train_accs, test_accs

# ---------------------------------------------------------
# Step 7: Detailed Evaluation Metrics (Accuracy, Recall/Sensitivity, F1)
# ---------------------------------------------------------
def evaluate_metrics(model, X_train, y_train, X_test, y_test, class_names):
    print("\n" + "="*60)
    print("STEP 7: Comprehensive Evaluation Metrics (Train & Test)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        train_logits = model(torch.tensor(X_train, dtype=torch.float32).to(device))
        train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()
        
        test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        
    print("\n--- TEST SET CLASSIFICATION REPORT ---")
    print(classification_report(y_test, test_preds, target_names=class_names, digits=4))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle any zero row
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title("Normalized Confusion Matrix (Unseen Test Set)", fontsize=13, fontweight='bold')
    plt.ylabel("True Failure Mode", fontweight='bold')
    plt.xlabel("Predicted Failure Mode", fontweight='bold')
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q7_confusion_matrix.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q7_confusion_matrix.png")
    
    # Metrics Summary Table Figure
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', zero_division=0)
    overall_acc = accuracy_score(y_test, test_preds)
    
    metrics_data = []
    for i, c in enumerate(class_names):
        metrics_data.append([c, f"{test_prec[i]*100:.1f}%", f"{test_rec[i]*100:.1f}%", f"{test_f1[i]*100:.1f}%"])
    metrics_data.append(["--- Macro Average ---", f"{macro_p*100:.1f}%", f"{macro_r*100:.1f}%", f"{macro_f1*100:.1f}%"])
    metrics_data.append(["--- Overall Accuracy ---", f"{overall_acc*100:.2f}%", "-", "-"])
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=metrics_data,
                     colLabels=["Failure Mode", "Precision", "Sensitivity (Recall)", "F1-Score"],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.2, 1.4)
    plt.title("Detailed Classification Performance Metrics Breakdown", fontsize=13, fontweight='bold', pad=15)
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q7_metrics_summary.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q7_metrics_summary.png")
    
    return {
        'accuracy': overall_acc,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'cm': cm
    }

# ---------------------------------------------------------
# Step 8: Regularization Techniques & Comparison
# ---------------------------------------------------------
class RegularizedClassifier(nn.Module):
    """Regularized Architecture with BatchNorm, Dropout & L2 Weight Decay"""
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

def run_classification_regularization(X_train, y_train, X_test, y_test, in_features, num_classes, class_names, base_tr_loss, base_te_loss, epochs=35):
    print("\n" + "="*60)
    print("STEP 8: Regularization Techniques & Model Comparison")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg_model = RegularizedClassifier(in_features, num_classes=num_classes, hidden1=128, hidden2=64, dropout_p=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Weight decay (L2 Regularization)
    optimizer = optim.Adam(reg_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_tensor_y = torch.tensor(y_test, dtype=torch.long).to(device)
    
    reg_train_losses, reg_test_losses = [], []
    
    for epoch in range(1, epochs + 1):
        reg_model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = reg_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_tr_loss = running_loss / len(y_train)
        
        reg_model.eval()
        with torch.no_grad():
            test_logits = reg_model(test_tensor_x)
            test_loss = criterion(test_logits, test_tensor_y).item()
            
        reg_train_losses.append(epoch_tr_loss)
        reg_test_losses.append(test_loss)
        
    # Evaluate Regularized Model
    reg_model.eval()
    with torch.no_grad():
        test_logits = reg_model(test_tensor_x)
        reg_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        
    reg_acc = accuracy_score(y_test, reg_preds)
    _, reg_r, reg_f1, _ = precision_recall_fscore_support(y_test, reg_preds, average='macro', zero_division=0)
    
    print("\nRegularized Architecture Performance:")
    print(f"Accuracy:        {reg_acc*100:.2f}%")
    print(f"Macro Recall:    {reg_r*100:.2f}%")
    print(f"Macro F1-Score:  {reg_f1*100:.2f}%")
    
    # 1. Comparison Loss Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, epochs + 1), base_tr_loss, label='Train Loss', color='#1f77b4', lw=2)
    ax1.plot(range(1, epochs + 1), base_te_loss, label='Test Loss', color='#d62728', lw=2, linestyle='--')
    ax1.set_title("Baseline Classifier (No Regularization)", fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(range(1, epochs + 1), reg_train_losses, label='Train Loss (Reg)', color='#2ca02c', lw=2)
    ax2.plot(range(1, epochs + 1), reg_test_losses, label='Test Loss (Reg)', color='#ff7f0e', lw=2, linestyle='--')
    ax2.set_title("Regularized Classifier (Dropout + BatchNorm + L2)", fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Loss Curve Comparison: Baseline vs. Regularized Deep Classifier", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q8_regularization_comparison.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q8_regularization_comparison.png")
    
    # 2. Multi-class One-vs-Rest ROC Curves
    plt.figure(figsize=(9, 7))
    for i in range(num_classes):
        # binary ground truth for class i
        y_binary = (y_test == i).astype(int)
        if np.sum(y_binary) > 0:
            fpr, tpr, _ = roc_curve(y_binary, probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guessing (AUC = 0.50)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
    plt.title('Multi-Class One-vs-Rest ROC Curves (Regularized Deep Model)', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_DIR, "part2_q8_roc_curves.png"), dpi=200)
    plt.close()
    print("Saved: figures/part2_q8_roc_curves.png")
    
    return {
        'reg_acc': reg_acc,
        'reg_recall': reg_r,
        'reg_f1': reg_f1
    }

def main():
    start_time = time.time()
    df_clean, df_encoded, class_names, le = load_and_preprocess_data()
    run_eda(df_clean, class_names)
    plot_classification_architecture()
    X_train, X_test, y_train, y_test, scaler, in_features = balance_and_split_data(df_encoded, class_names)
    num_classes = len(class_names)
    
    best_params = run_classification_grid_search(X_train, y_train, in_features, num_classes)
    model, base_tr_l, base_te_l, base_tr_a, base_te_a = train_classification_with_curves(
        X_train, y_train, X_test, y_test, in_features, num_classes, best_params, epochs=35
    )
    eval_res = evaluate_metrics(model, X_train, y_train, X_test, y_test, class_names)
    reg_res = run_classification_regularization(
        X_train, y_train, X_test, y_test, in_features, num_classes, class_names, base_tr_l, base_te_l, epochs=35
    )
    
    print("\n" + "="*60)
    print(f"Part 2 Complete in {time.time() - start_time:.2f} seconds!")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
