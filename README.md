# Telugu Fake Review Detection

A machine learning pipeline to detect fake reviews written in **Telugu** (a South Indian language), using three different NLP embedding strategies and multiple ML classifiers.

---

## 📌 Project Overview

With the rise of user-generated content on platforms like YouTube, fake and spam reviews have become a major problem — especially for regional languages like Telugu. This project builds an end-to-end pipeline that:

1. Scrapes Telugu comments from YouTube
2. Augments the dataset with synthetic fake reviews
3. Extracts features using three embedding models
4. Trains and evaluates ML classifiers to detect fake reviews

---

## 📁 Project Structure

```
Telugu/
│
├── Data_Extraction/
│   ├── telugu_comments_api.py        # Scrapes Telugu YouTube comments via YouTube Data API v3
│   ├── augment.py                    # Generates 700 synthetic fake reviews using Telugu templates
│   ├── feature_extraction.py         # Extracts 768-dim IndicBERT embeddings + quick SVM baseline
│   ├── resampling.py                 # Undersamples majority class to balance dataset
│   └── telugu_reviews_all.xlsx       # Raw scraped Telugu YouTube comments
│
├── ML_Models_Bert/
│   ├── ML_models_Telugu.ipynb        # KNN + Random Forest on IndicBERT features
│   ├── Telugu_BERT__ML_Models_PCA.ipynb  # PCA (100 components) + classification
│   ├── telugu_indicbert_balanced.xlsx    # Balanced IndicBERT feature set
│   └── telugu_indicbert_pca.xlsx         # PCA-reduced IndicBERT feature set
│
├── ML_Models_Fasttext/
│   ├── telugu_fasttext.ipynb             # FastText feature extraction (300-dim)
│   ├── Telugu_fasttext_ML_models.ipynb   # KNN + Random Forest on FastText features
│   ├── Telugu_fasttext_PCA_ML_Models.ipynb  # PCA + classification on FastText
│   ├── telugu_fasttext_features_output.xlsx  # Raw FastText features
│   ├── telugu_fasttext_pca.xlsx          # PCA-reduced FastText features
│   └── pca_variance.csv                  # Explained variance per PCA component
│
├── ML_Models_LASER/
│   ├── Laser_embedding.ipynb             # LASER embedding extraction (1024-dim)
│   ├── Telugu_LASER_ML_Models.ipynb      # KNN + Random Forest on LASER features
│   ├── telugu_reviews_with_laser.xlsx    # Raw LASER embeddings
│   ├── telugu_LASER_balanced.xlsx        # Balanced LASER feature set
│   └── telugu_LASER_pca.xlsx             # PCA-reduced LASER feature set
│
└── README.md
```

---

## 🧠 Embedding Strategies

| Embedding | Model | Dimensions | Best Accuracy |
|-----------|-------|-----------|---------------|
| IndicBERT | `ai4bharat/indic-bert` | 768 | ~87.2% |
| FastText | `cc.te.300.bin` (Facebook) | 300 | ~94.2% ✅ |
| LASER | `laserembeddings` (Meta) | 1024 | — |

---

## ⚙️ ML Pipeline

All three embedding approaches share the same classification pipeline:

- **Models:** K-Nearest Neighbors (KNN) and Random Forest
- **Tuning:** GridSearchCV with 5-fold cross-validation
- **Split:** 80% train / 20% test
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Optional:** PCA (100 components) for dimensionality reduction

---

## 🗂️ Data

- **Real reviews:** Telugu comments scraped from YouTube using the YouTube Data API v3
- **Fake reviews:** 700 synthetically generated using Telugu templates covering:
  - Extreme emotional language (84.5%)
  - Poor quality writing (14.6%)
  - Unusual ratings (0.9%)
- **Label convention:** `0` = Real, `1` = Fake

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/telugu-fake-review-detection.git
cd telugu-fake-review-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

**Step 1 — Collect data:**
```bash
python Data_Extraction/telugu_comments_api.py
```
> Add your YouTube Data API key in the script before running.

**Step 2 — Augment with fake reviews:**
```bash
python Data_Extraction/augment.py
```

**Step 3 — Extract features:**
```bash
python Data_Extraction/feature_extraction.py
```

**Step 4 — Balance the dataset:**
```bash
python Data_Extraction/resampling.py
```

**Step 5 — Train and evaluate models:**
Open and run the notebooks in `ML_Models_Bert/`, `ML_Models_Fasttext/`, or `ML_Models_LASER/` depending on the embedding you want to use.

---

## 📦 Dependencies

- `pandas`, `numpy`
- `transformers`, `torch` (IndicBERT)
- `fasttext` (FastText embeddings)
- `laserembeddings` (LASER embeddings)
- `scikit-learn` (ML models, PCA, GridSearchCV)
- `google-api-python-client` (YouTube Data API)
- `emoji`, `indic-transliteration`
- `matplotlib`, `seaborn`

---

## 📊 Results Summary

| Embedding | Model | Accuracy |
|-----------|-------|----------|
| IndicBERT | KNN | ~87.2% |
| IndicBERT + PCA | KNN | — |
| FastText | KNN | ~94.2% |
| FastText + PCA | KNN | — |
| LASER | KNN | — |

> FastText achieves the best accuracy, likely due to its strong word-level representations for morphologically rich languages like Telugu.

---

## 📝 Notes

- Notebooks were originally developed in **Google Colab**. File paths use `/content/drive/MyDrive/...` — update these to your local paths if running locally.
- The YouTube API key in `telugu_comments_api.py` is intentionally left blank. Add your own key from [Google Cloud Console](https://console.cloud.google.com/).

---

## 👤 Author

**Vignesh Kothuri**

---

## 📄 License

This project is licensed under the MIT License.
