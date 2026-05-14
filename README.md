# 🍷 Wine Quality Prediction

Predict wine quality from chemical characteristics using three machine learning classifiers: **Random Forest**, **Stochastic Gradient Descent (SGD)**, and **Support Vector Classifier (SVC)**.

---

## 📁 Project Structure

```
wine-quality-prediction/
├── data/
│   └── WineQT.csv               # Dataset (1143 wine samples, 11 features)
├── notebooks/
│   └── wine_quality_prediction.ipynb  # Full end-to-end analysis notebook
├── src/
│   ├── train.py                 # Training script (CLI)
│   └── predict.py               # Inference script (CLI)
├── models/                      # Saved model .pkl files (generated after training)
├── visualizations/              # Generated plots (generated after training)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| Samples | 1,143 |
| Features | 11 chemical attributes |
| Target | Quality score (3 – 8) |

### Features

| Feature | Description |
|---|---|
| `fixed acidity` | Tartaric acid concentration |
| `volatile acidity` | Acetic acid concentration (vinegar-like) |
| `citric acid` | Freshness & flavour enhancer |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt content |
| `free sulfur dioxide` | Free SO₂ (microbial protection) |
| `total sulfur dioxide` | Total SO₂ |
| `density` | Density of wine |
| `pH` | Acidity/alkalinity |
| `sulphates` | Antimicrobial additive |
| `alcohol` | Alcohol percentage |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

### Option A — Jupyter Notebook (recommended for exploration)

```bash
jupyter notebook notebooks/wine_quality_prediction.ipynb
```

Run all cells to load data, visualise it, train all three models, and compare results.

### Option B — Python Script (command line)

```bash
python src/train.py
# or with custom paths:
python src/train.py --data data/WineQT.csv --output models/ --vis visualizations/
```

This will:
- Train all three classifiers
- Print accuracy & classification reports to the console
- Save model `.pkl` files to `models/`
- Save all plots to `visualizations/`

---

## 🔮 Prediction

After training, run inference on new data:

```bash
python src/predict.py \
    --model  models/random_forest_classifier.pkl \
    --scaler models/scaler.pkl \
    --input  data/WineQT.csv
```

The input CSV should contain the same 11 chemical feature columns as the training data.

---

## 🤖 Models

| Model | Strengths | Scaling Required |
|---|---|---|
| **Random Forest** | Handles non-linearity, robust to outliers, provides feature importances | No |
| **SGD Classifier** | Extremely fast, suitable for large datasets | Yes |
| **SVC (RBF kernel)** | Strong in high-dimensional space, great generalisation | Yes |

### Hyperparameters Used

```python
RandomForestClassifier(n_estimators=200, random_state=42)
SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
```

---

## 📈 Key Visualisations Generated

| File | Description |
|---|---|
| `quality_distribution.png` | Bar chart of quality score counts |
| `correlation_heatmap.png` | Feature-to-feature correlation matrix |
| `features_vs_quality.png` | Boxplots of key chemical features by quality |
| `feature_distributions.png` | Histogram of every feature |
| `feature_importances.png` | Random Forest feature importance ranking |
| `cm_random_forest_classifier.png` | Confusion matrix — Random Forest |
| `cm_sgd_classifier.png` | Confusion matrix — SGD |
| `cm_svc_rbf_kernel.png` | Confusion matrix — SVC |
| `model_comparison.png` | Side-by-side accuracy bar chart |

---

## 🔑 Key Findings

- **Alcohol**, **volatile acidity**, **sulphates**, and **density** are the strongest predictors of wine quality.
- Quality scores are heavily concentrated at 5 and 6 (out of 3–8), creating a mild class imbalance.
- Random Forest consistently outperforms the other classifiers on this dataset.

---

## 🛠️ Potential Improvements

- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Handle class imbalance with **SMOTE** oversampling
- Try gradient boosting methods: **XGBoost**, **LightGBM**
- Cross-validation for more robust accuracy estimates
- Deploy as a REST API with **FastAPI** or **Flask**

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation & analysis |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, metrics |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `joblib` | Model serialisation |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
