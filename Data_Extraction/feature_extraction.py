import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Step 1: Load the merged dataset
input_file = "telugu_reviews_merged.xlsx"
try:
    df = pd.read_excel(input_file)
except FileNotFoundError:
    print(f"Error: File '{input_file}' not found. Please check the file path.")
    exit(1)
except ImportError:
    print("Error: 'pandas' or 'openpyxl' not installed. Run 'pip install pandas openpyxl'.")
    exit(1)

# Verify columns
expected_columns = ['comment', 'preprocessed_comment', 'label', 'reason']
if not all(col in df.columns for col in expected_columns):
    print(f"Error: Missing expected columns: {expected_columns}")
    exit(1)

print(f"Dataset size: {len(df)} reviews")
print(f"Real reviews (Label 0): {len(df[df['label'] == 0])}")
print(f"Fake reviews (Label 1): {len(df[df['label'] == 1])}")

# Extract texts and labels
texts = df['comment'].values  # Use original comment for IndicBERT
labels = df['label'].values   # 0 for real, 1 for fake

# Step 2: Extract IndicBERT Embeddings
def extract_indicbert_embeddings(texts, batch_size=16):
    try:
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        model = AutoModel.from_pretrained("ai4bharat/indic-bert")
    except ImportError:
        print("Error: 'transformers' or 'sentencepiece' not installed. Run 'pip install transformers sentencepiece'.")
        exit(1)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts.tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
        
        print(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
    
    embeddings = np.vstack(embeddings)
    print(f"IndicBERT embeddings shape: {embeddings.shape}")
    return embeddings

# Generate embeddings
try:
    indicbert_embeddings = extract_indicbert_embeddings(texts)
except Exception as e:
    print(f"Error during embedding extraction: {e}")
    exit(1)

# Step 3: Create DataFrame for features
# Create column names for 768 features
feature_columns = [f"feature_{i}" for i in range(indicbert_embeddings.shape[1])]
# Combine embeddings, labels, and comments into a DataFrame
features_df = pd.DataFrame(
    indicbert_embeddings,
    columns=feature_columns
)
features_df['label'] = labels
features_df['comment'] = df['comment']  # Optional: include for reference

# Step 4: Save to new Excel file
output_file = "telugu_indicbert_features.xlsx"
try:
    features_df.to_excel(output_file, index=False)
    print(f"Features saved to '{output_file}'")
except Exception as e:
    print(f"Error saving Excel file: {e}")
    exit(1)

# Step 5: Train and Evaluate with SVM
try:
    X = indicbert_embeddings
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print("\nSVM Classification Report (IndicBERT Embeddings):")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
except ImportError:
    print("Error: 'scikit-learn' not installed. Run 'pip install scikit-learn'.")
    exit(1)
except Exception as e:
    print(f"Error during SVM training: {e}")
    exit(1)