# 💳 Credit Card Fraud Detection with Isolation Forest

This project applies unsupervised anomaly detection using **Isolation Forest** on the popular `creditcard.csv` dataset to detect fraudulent transactions.

---

## 📊 Project Overview

- Dataset: `creditcard.csv` (Kaggle)
- Goal: Detect fraudulent transactions using unsupervised learning
- Technique: Isolation Forest (unsupervised anomaly detection)
- Key Steps:
  - Feature selection using correlation
  - Z-score standardization
  - Visualization of anomalies
  - Model evaluation on both individual features and full feature set

---

## 🔍 Feature Selection

1. **Correlation with Target (`Class`)**:
   - Selected features with strong correlation > 0.2
   - Picked: `V10`, `V12`, `V14`, `V17`

2. **Visualization**:
   - Plotted correlation bar chart
   - Highlighted anomalous vs. normal data using color-coded scatter plots for each selected feature

---

## ⚙️ Preprocessing & Anomaly Detection

- Used **Z-Score Normalization** with `StandardScaler`
- Ran Isolation Forest on:
  - Individual features (for visualization)
  - Entire feature set (for training and evaluation)

---

## 🧪 Model Training & Evaluation

- Splitting data (80/20 train-test split)
- Applied `IsolationForest` with `contamination=0.005` and `n_estimators=200`
- Converted prediction labels (-1 → 0, 1 → 1)

---

## 📈 Results

| Metric      | Training Set | Test Set   |
|-------------|--------------|------------|
| **Accuracy**   | 0.9962       | 0.9960     |
| **Precision**  | 0.9965       | 0.9997     |
| **Recall**     | 0.9997       | 0.9963     |
| **F1 Score**   | 0.9981       | 0.9980     |

> ✅ The model achieved **excellent performance** in detecting anomalies despite being trained in an unsupervised manner.

---

## 🛠️ Tools & Libraries

- `pandas`, `numpy`
- `scikit-learn`: `IsolationForest`, metrics, preprocessing
- `matplotlib` , 'seaborn'for plotting

---

## 🚀 How to Run

1. Clone the repository
2. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
3. Run the Python script:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python isolation_forest_fraud.py

