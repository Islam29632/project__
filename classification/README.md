🧠 Classification Projects
1. 📊 Digits Classification Using Support Vector Machines (SVC)
  In this project, I implemented a multi-class classification system to recognize handwritten digits (0–9) from sklearn.datasets using a Support Vector Classifier (SVC)    in Python.
  Key steps included:

      Preprocessing: Visualized data examples for clarity.

      Model Training: Optimized hyperparameters (kernel, C, gamma) via grid search.

      Evaluation:
      Achieved an F1-score of 0.967, precision of 0.967, and recall of 0.967, demonstrating robust performance in balancing false positives/negatives.

      This project sharpened my skills in model interpretability, metric selection, and handling high-dimensional data—critical for real-world AI applications like OCR         or fraud detection.

2. 🌸 Iris Flower Classification: Comparing Logistic Regression, KNN, and Decision Trees
    In this project, I built a multi-class classifier to predict Iris flower species (Setosa, Versicolor, Virginica) using Logistic Regression, K-Nearest Neighbors           (KNN), and Decision Trees.
   
    Key steps included:

      Exploratory Data Analysis (EDA): Visualized feature distributions (sepal/petal dimensions) to identify class separability.

    Model Training & Tuning:

      Tuned hyperparameters (e.g., max_depth for Decision Trees, n_neighbors for KNN).

    Evaluation:
      Achieved ~97% accuracy with Logistic Regression, highlighting its efficiency for linearly separable data. KNN and Decision Trees provided insights into non-linear        boundaries.

This project strengthened my understanding of algorithm selection, hyperparameter tuning, and interpretability—skills directly applicable to real-world classification tasks like medical diagnosis or customer segmentation.

3. 🍷 Wine Dataset Clustering: PCA-Driven Feature Optimization & Model Evaluation
    In this project, I analyzed the Wine dataset (from scikit-learn) to explore intrinsic patterns in wine chemical properties using unsupervised learning.
    Key steps included:

  🔹 Pre-PCA Clustering:
      Applied K-Means and Hierarchical Clustering on raw features.

      Evaluated clustering performance using:

      Silhouette Score: 0.285 (K-Means), 0.277 (Hierarchical)

      Davies-Bouldin Index: 1.389 (K-Means), 1.419 (Hierarchical)
      → Indicating suboptimal separation due to high dimensionality.

  🔹 PCA-Driven Optimization:
      Applied Principal Component Analysis (PCA) to retain 95% variance for clearer feature interpretation.

      Re-applied clustering algorithms with significantly improved results:

      K-Means: Silhouette ↑ to 0.454, Davies-Bouldin ↓ to 0.839

      Hierarchical: Silhouette ↑ to 0.446, Davies-Bouldin ↓ to 0.852

This project honed my skills in dimensionality reduction, cluster validation, and metric-driven model selection—essential for applications like customer segmentation or anomaly detection.
