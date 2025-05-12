# 🧪 Glass Type Prediction using Machine Learning

## 📘 Project Overview

This project aims to classify types of glass based on their chemical properties using various machine learning models. The dataset used is the **Glass Identification Dataset** from the UCI Machine Learning Repository.

The primary objective is to compare the performance of different classifiers such as:
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (Neural Network)

---

## 📂 Dataset Information

- **Source:** [UCI Machine Learning Repository – Glass Identification Dataset](https://archive.ics.uci.edu/ml/datasets/glass+identification)
- **Features:** 9 chemical attributes (e.g., RI, Na, Mg, etc.)
- **Target:** Glass type (multi-class classification)

---

## 🔧 Methodology

1. **Data Preprocessing**
   - Checked for missing values
   - Standardized features using `StandardScaler`

2. **Dimensionality Reduction**
   - Applied PCA to retain 95% variance

3. **Modeling**
   - Trained five classifiers using stratified train-test split (80-20)
   - Evaluated using classification report, confusion matrix, and ROC curves

4. **Visualization**
   - Feature distribution, boxplots, and correlation heatmaps
   - Model-wise ROC curves and accuracy comparison bar chart

---

## 📊 Model Performance

| Model                 | Accuracy |
|----------------------|----------|
| Random Forest         | ~0.93    |
| SVM                   | ~0.88    |
| Neural Network (MLP)  | ~0.86    |
| KNN                   | ~0.84    |
| Decision Tree         | ~0.82    |

> 📌 *Note: Accuracy values may vary slightly due to random state and data splits.*

---

## 📈 Visualizations Included

- Correlation heatmap of features
- Boxplot for outlier detection
- Confusion matrices for each model
- ROC curves (multi-class)
- Accuracy comparison chart

---

## 💻 Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `ucimlrepo`

Install them using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

---

## 🚀 How to Run

```python
# Step 1: Fetch the dataset
from ucimlrepo import fetch_ucirepo

# Step 2: Preprocess and analyze
# (Check the Jupyter Notebook for complete pipeline)

# Step 3: Train and evaluate models
# Models include: DecisionTree, RandomForest, SVM, KNN, MLP
```

---

## 📄 Output

A PDF report `Glass Type Prediction Report.pdf` is attached, detailing the model results, visualizations, and insights.

---

