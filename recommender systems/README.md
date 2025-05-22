# 📚 Recommendation Systems Portfolio

This repository showcases **three different recommendation systems** implemented using **MovieLens** and **Goodreads** datasets. Each project demonstrates various techniques such as **Exploratory Data Analysis (EDA)**, **Content-Based Filtering**, **Collaborative Filtering**, and evaluation metrics like **MAE, RMSE, Precision, and Recall**.

---

## 📁 Projects Overview

### 1. 🎥 MovieLens 20M Dataset — Hybrid Recommendation System

**Techniques:**
- EDA on genres, movies, and user ratings  
- Content-Based Filtering  
- User-Based Collaborative Filtering  
- Evaluation with MAE & RMSE

**Key Steps:**
- Load and merge `ratings.csv` and `movies.csv`
- Visualize:
  - Top 10 genres
  - Most rated movies
  - Ratings per user
- Generate recommendations using a hybrid approach (user + content)
- Evaluate performance:
  - **MAE**: `0.846`
  - **RMSE**: `1.060`

---

### 2. 🎬 MovieLens Latest Small — Collaborative Filtering

**Techniques:**
- EDA & user-item matrix creation  
- User Similarity with Cosine Similarity  
- Weighted Average Ratings from similar users  
- User-Based and Item-Based predictions  
- Evaluation with MAE, Precision, Recall

**Key Steps:**
- Build a pivot table (user-item matrix)
- Calculate similarity between users
- Predict ratings using a weighted average from top similar users
- Evaluate model:
  - **MAE (User-Based)**: ~
  - **MAE (Item-Based)**: ~
  - **Mean Precision**: `0.2056`
  - **Mean Recall**: `0.2284`

---

### 3. 📖 Goodreads Book Recommendation — Content-Based Filtering

**Techniques:**
- EDA on most popular authors & genres  
- TF-IDF + Cosine Similarity for recommendations  
- Evaluation using Precision, Recall, and F1-Score

**Key Steps:**
- Visualize:
  - Top authors
  - Most frequent genres
- Generate recommendations using genre-based content filtering
- Evaluate with a test genre set:
  - Compute **precision**, **recall**, and **F1-score**
  - Example: Recommend top 5 books for top 10 genres

---

## 🧪 Evaluation Summary

| Project              | Method(s) Used                             | Metrics Summary                                      |
|----------------------|--------------------------------------------|------------------------------------------------------|
| MovieLens 20M        | Hybrid (Content + User CF)                 | MAE: 0.846, RMSE: 1.060                              |
| MovieLens Small      | Collaborative Filtering (User & Item)      | MAE (User/Item): ~, Precision: 0.2056, Recall: 0.2284|
| Goodreads            | Content-Based (Genre + TF-IDF + Cosine)    | Evaluated using Precision, Recall, F1                |

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn
- Cosine Similarity, TF-IDF
- Surprise or custom CF implementations
