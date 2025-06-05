# Stroke Prediction & Patient Risk Clustering

This project applies machine learning and unsupervised learning techniques to predict stroke events and uncover hidden patterns in patient health data. The goal is to support early diagnosis and better understand risk groups through clustering.

## Dataset

It contains medical and demographic data for over 5,000 individuals, including whether they have had a stroke.

## Features

- **Target Variable**: `stroke`
- **Input Features**: age, hypertension, heart disease, marital status, work type, residence type, avg glucose level, BMI, smoking status, etc.

## Preprocessing

- Removed duplicates and fixed invalid entries
- Imputed missing values (notably in BMI) using mean substitution
- One-hot encoded categorical features
- Addressed class imbalance with **SMOTE** (Synthetic Minority Over-sampling Technique)

## Machine Learning Models

- Applied and evaluated several classifiers:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **SVM (RBF Kernel)**
  - **KNN**
  - **Naive Bayes**
- **Hyperparameter tuning** performed with **Optuna**
- **Best Result**: SVM (RBF) with 97.3% accuracy

## Dimensionality Reduction & Visualization

- Used **PCA** , **LDA** and **t-SNE** to reduce data dimensionality and visualize class separation in 2D and 3D
- Revealed strong separation between stroke and non-stroke cases

## Clustering Analysis

- Applied **K-Means** and **Hierarchical Clustering** to group patients
- Identified clusters with high stroke prevalence
- Visualized clusters using PCA and 3D plots to show separation and stroke distribution

## Results Summary

| Model           | Accuracy | AUC   |
|----------------|----------|-------|
| Logistic Reg.   | 94.2%    | 0.93  |
| Decision Tree   | 93.4%    | 0.92  |
| **SVM (RBF)**    | **97.3%**  | **0.97**  |
| Random Forest   | 95.8%    | 0.95  |
| KNN             | 95.6%    | 0.96  |
| Naive Bayes     | 93.7%    | 0.93  |

## Technologies Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Optuna (for hyperparameter tuning)
- Plotly (for 3D visualization)
