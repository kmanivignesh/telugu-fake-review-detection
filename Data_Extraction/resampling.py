import pandas as pd
from sklearn.utils import resample

# 1. Load the full dataset
df = pd.read_excel('telugu_indicbert_features.xlsx')  # Adjust path if needed

# 2. Inspect column names
print("Columns:", df.columns[-5:])  # last 5 columns, to confirm label and comment
print("Class distribution:\n", df['label'].value_counts())

# 3. Split by class
df_real = df[df['label'] == 0]
df_fake = df[df['label'] == 1]

# 4. Undersample majority class
min_count = min(len(df_real), len(df_fake))
df_real_downsampled = resample(df_real, replace=False, n_samples=min_count, random_state=42)
df_fake_downsampled = resample(df_fake, replace=False, n_samples=min_count, random_state=42)

# 5. Combine and shuffle
df_balanced = pd.concat([df_real_downsampled, df_fake_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Save to new Excel
df_balanced.to_excel('telugu_indicbert_balanced.xlsx', index=False)

print("✅ Balanced dataset saved as 'telugu_indicbert_balanced.xlsx'")
print("New class distribution:\n", df_balanced['label'].value_counts())
