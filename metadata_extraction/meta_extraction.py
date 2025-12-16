import pandas as pd
import re

# Load metadata Excel
df = pd.read_excel("metadata.xlsx")

# Normalize subject_id (IMPORTANT)
def normalize_subject_id(x):
    x = str(x)
    m = re.search(r'\d{7,8}', x)
    return m.group(0) if m else x

df['subject_id'] = df['subject id'].apply(normalize_subject_id)

# Map diagnosis to binary labels
df['label'] = df['type'].map({
    'MDD': 1,
    'HC': 0
})

# Keep only required columns
metadata = df[['subject_id', 'label']]

# Safety check
print(metadata['label'].value_counts())
print(metadata.head())

# Save for Notebook 3
metadata.to_csv("metadata.csv", index=False)

print("✅ metadata.csv created successfully")