import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load joblib files
print("Loading joblib files...")

preprocessor = joblib.load("preprocessor.joblib")
lr_model = joblib.load("model_LogisticRegression.joblib")
rf_model = joblib.load("model_RandomForest.joblib")

print("All models loaded successfully.\n")

# 2. Extract transformer safely
if isinstance(preprocessor, dict):
    transformer = preprocessor.get("scaler") or preprocessor.get("preprocessor")
else:
    transformer = preprocessor

# 3. Get correct feature names
if hasattr(transformer, "feature_names_in_"):
    feature_names = transformer.feature_names_in_
    print("Using trained feature names:")
    print(feature_names)
else:
    raise RuntimeError(
        "❌ Cannot find feature names. "
        "Preprocessor was likely trained on NumPy array."
    )

# 4. Create input using REAL feature names

# Dummy values for testing (replace with real values later)
sample_values = np.random.rand(1, len(feature_names))
sample_data = pd.DataFrame(sample_values, columns=feature_names)

print("\nInput data:")
print(sample_data)

# 5. Apply preprocessing
print("\nApplying preprocessing...")
X_processed = transformer.transform(sample_data)

# 6. Predictions
lr_pred = lr_model.predict(X_processed)
rf_pred = rf_model.predict(X_processed)

lr_prob = lr_model.predict_proba(X_processed)[0]
rf_prob = rf_model.predict_proba(X_processed)[0]

print("\nPredictions:")
print(f"Logistic Regression : {lr_pred[0]}")
print(f"Random Forest       : {rf_pred[0]}")

print("\nProbabilities:")
print("LR:", lr_prob)
print("RF:", rf_prob)

# 7. Plot probabilities
labels = ["Class 0", "Class 1"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, lr_prob, width, label="Logistic Regression")
plt.bar(x + width/2, rf_prob, width, label="Random Forest")

plt.xlabel("Classes")
plt.ylabel("Probability")
plt.title("Prediction Probability Comparison")
plt.xticks(x, labels)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig("model_probability_comparison.png", dpi=300)
print("Plot saved as model_probability_comparison.png")