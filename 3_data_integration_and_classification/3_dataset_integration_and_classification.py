import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sns.set_theme()

# ==========================================
# CONFIGURATION
# ==========================================
# Input folder (where your feature CSVs are)
INPUT_DIR = "data_processed" 

# Output folder (Where the Dashboard expects models)
# We point this to the folder your Streamlit app uses
OUTPUT_MODEL_DIR = "../Model_Output_Joblib" 
FINAL_DATA_PATH = "data_processed/MODMA_multimodal_features.csv"

# Ensure output directories exist
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def normalize_subject_id(x):
    """Fixes '2010001.0' -> '2010001' mismatches"""
    if pd.isna(x):
        return x
    x = str(x)
    # Remove .0 ending if present
    x = re.sub(r'\.0$', '', x)
    # Extract the main ID pattern
    m = re.search(r'\d{7,8}', x)
    return m.group(0) if m else x

# ==========================================
# 1. LOAD FEATURES
# ==========================================
feature_files = {
    'audio': os.path.join(INPUT_DIR, "audio_features.csv"),
    'eeg3': os.path.join(INPUT_DIR, "eeg_3ch_features.csv"),
    'eeg128': os.path.join(INPUT_DIR, "eeg_resting_128ch_features.csv"),
    'erp': os.path.join(INPUT_DIR, "eeg_erp_features.csv"),
}

loaded_dfs = []

for name, path in feature_files.items():
    if not os.path.exists(path):
        print(f"⚠️  Warning: File not found: {path} (Skipping {name})")
        continue

    df = pd.read_csv(path)
    
    # Normalize IDs immediately
    if 'subject_id' in df.columns:
        df['subject_id'] = df['subject_id'].apply(normalize_subject_id)
        loaded_dfs.append(df)
        print(f"✅ Loaded {name}: {df.shape}")
    else:
        print(f"❌ Error: {name} missing 'subject_id' column")

if not loaded_dfs:
    raise ValueError("❌ No feature files were loaded! Check your 'data_processed' folder.")

# ==========================================
# 2. MERGE DATA
# ==========================================
# Safer merge strategy: Reduce list of DFs
df_merge = reduce(lambda left, right: pd.merge(left, right, on='subject_id', how='outer'), loaded_dfs)

print(f"📊 Shape after raw merge: {df_merge.shape}")

# Keep numeric features only (preserving subject_id)
numeric_cols = df_merge.select_dtypes(include=[np.number]).columns.tolist()
if 'subject_id' not in numeric_cols:
    cols_to_keep = ['subject_id'] + numeric_cols
else:
    cols_to_keep = numeric_cols

# Group by subject_id to aggregate duplicates (Taking Mean)
df_merge = df_merge.groupby('subject_id', as_index=False)[numeric_cols].mean()

print(f"📊 Shape after subject aggregation: {df_merge.shape}")

# Save the consolidated features CSV (This is what the Dashboard loads)
df_merge.to_csv(FINAL_DATA_PATH, index=False)
print(f"💾 Saved merged dataset to: {FINAL_DATA_PATH}")

# ==========================================
# 3. MERGE LABELS
# ==========================================
if not os.path.exists("metadata.csv"):
    raise FileNotFoundError("❌ metadata.csv not found (required for labels)")

meta = pd.read_csv("metadata.csv")
meta['subject_id'] = meta['subject_id'].apply(normalize_subject_id)

# Inner merge to keep only labeled data for training
df_labeled = df_merge.merge(meta[['subject_id', 'label']], on='subject_id', how='inner')

print(f"📊 Shape after adding labels: {df_labeled.shape}")
print(f"Proportion:\n{df_labeled['label'].value_counts()}")

if df_labeled.empty:
    raise ValueError("❌ Dataset is empty after merging labels! Check subject_id formats in metadata vs features.")

# ==========================================
# 4. PREPROCESSING
# ==========================================
X = df_labeled.drop(columns=['subject_id', 'label'])
y = df_labeled['label'].astype(int).values

# Remove columns that are completely empty (NaN)
X = X.dropna(axis=1, how='all')
valid_cols = X.columns.tolist()

print(f"✨ Features remaining after cleaning: {len(valid_cols)}")

# Imputation & Scaling
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Fit on ALL labeled data (to save robust preprocessors)
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=valid_cols)
X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=valid_cols)

# SAVE PREPROCESSORS (Crucial for Dashboard)
# Using 'scaler' key to match Dashboard expectations
joblib.dump(
    {'imputer': imputer, 'scaler': scaler},
    os.path.join(OUTPUT_MODEL_DIR, "preprocessor.joblib")
)
print(f"💾 Saved preprocessor to {OUTPUT_MODEL_DIR}")

# ==========================================
# 5. MODEL TRAINING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=2000)
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Save Model
    save_path = os.path.join(OUTPUT_MODEL_DIR, f"model_{name}.joblib")
    joblib.dump(model, save_path)
    
    # Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✅ Accuracy: {acc:.3f}")
    print(f"   💾 Saved to: {save_path}")

# ==========================================
# 6. FEATURE IMPORTANCE (VIZ)
# ==========================================
rf = models["RandomForest"]
if hasattr(rf, "feature_importances_"):
    importances = pd.Series(rf.feature_importances_, index=valid_cols).sort_values(ascending=False).head(20)
    
    print("\nTop 5 Features:")
    print(importances.head(5))
    
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_MODEL_DIR, "feature_importance.png"))
    print("📊 Feature importance plot saved.")

print("\n✅ Training Complete. Restart your Streamlit app now.")