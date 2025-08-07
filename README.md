# 📘 Machine Learning Complete Course – Jupyter Notebooks

Welcome to this curated repository of Jupyter notebooks that cover the **entire journey of a Machine Learning course**, from foundational topics to advanced concepts. Designed to be practical and beginner-friendly, this collection is ideal for students, data science learners, and professionals revising their ML concepts.

---

## 🔥 What You'll Find Here

- ✅ Expert-level explanations in clean, well-structured Jupyter notebooks  
- ✅ Real-world examples with datasets  
- ✅ Modular learning (one topic per notebook)  
- ✅ 7-Day Crash Courses for focused learning  
- ✅ Final advanced concepts like Explainable AI and Model Interpretation

---

## 📅 7-Day Mini Courses (with Projects)

These are **fast-track, structured 7-day tracks** where each day focuses on mastering one key concept. Ideal for those who want to specialize quickly.

### 🔷 7-Day Course: Data Transformation

Learn how to clean, encode, and prepare real-world messy datasets for machine learning models.

- Day 1: Introduction to messy data and common problems (nulls, mixed types)
- Day 2: Handling missing values (mean, median, KNN imputation)
- Day 3: Label encoding vs One-hot encoding
- Day 4: Scaling and normalization (MinMax, StandardScaler)
- Day 5: Binning, log transform, box-cox transform
- Day 6: Feature engineering: date, text, categorical split
- Day 7: Final project – Clean and transform Titanic dataset end-to-end

---

### 🔷 7-Day Course: Data Reduction

Learn how to simplify large datasets without losing performance using dimensionality reduction.

- Day 1: Introduction to curse of dimensionality
- Day 2: Feature selection techniques (filter, wrapper, embedded)
- Day 3: Correlation heatmaps and VIF
- Day 4: PCA theory and implementation
- Day 5: t-SNE for non-linear reduction
- Day 6: Autoencoders for compression
- Day 7: Final project – Reduce features of a real dataset and visualize results

---

### 🔷 7-Day Course: Data Integration

Learn how to combine data from multiple formats like CSV, JSON, and logs for a unified ML pipeline.

- Day 1: What is data integration? Why needed?
- Day 2: Reading multiple CSVs and merging
- Day 3: Parsing and flattening JSON into DataFrames
- Day 4: Processing log data (web traffic, sensor logs)
- Day 5: Joining tables: inner, outer, left, right joins
- Day 6: Schema matching and fixing mismatched fields
- Day 7: Final project – Merge student info (CSV), test scores (JSON), and website logs

---

## 🧠 Machine Learning Algorithms (with Practical Logic)

Here’s what you'll learn and implement in full detail:

### ✅ Exploratory Data Analysis (EDA)
Understand your data before modeling. Covers missing values, outliers, correlation, feature types, and visualization.

### ✅ Decision Trees
- Understand splitting using Gini/Entropy
- Visualize how decisions are made
- Deal with overfitting using pruning
- Real implementation using `sklearn.tree.DecisionTreeClassifier`

### ✅ Support Vector Machines (SVM)
- Learn how SVM finds optimal hyperplanes
- Kernel tricks (linear, RBF, polynomial)
- Tuning `C`, `gamma`, and `kernel` to control margin and overfitting
- Visual decision boundaries on 2D datasets

### ✅ Ensemble Learning
- Bagging: Random Forest (multiple decision trees)
- Boosting: GradientBoost, AdaBoost, XGBoost
- Learn how ensemble models reduce bias/variance
- Feature importances and model comparison

### ✅ K-Means Clustering
- Unsupervised learning – No labels required
- Elbow method and inertia
- Visualizing clusters
- Limitations (spherical assumption, sensitivity to `k`)

### ✅ Hierarchical Clustering
- Bottom-up (agglomerative) approach
- Dendrogram interpretation
- Linkage criteria: single, complete, average
- Use cases in genetics, customer segmentation

### ✅ Principal Component Analysis (PCA)
- Learn how PCA reduces dimensionality using linear algebra
- Eigenvalues/eigenvectors explained
- Visualize original vs reduced dimensions
- Keep 95% variance in fewer dimensions

### ✅ Class Imbalance Handling
- Use SMOTE, ADASYN, and random oversampling
- ROC-AUC, Precision-Recall instead of just Accuracy
- Real use case: fraud detection with imbalance

### ✅ Cross Validation & Hyperparameter Tuning
- Implement k-fold, stratified k-fold CV
- GridSearchCV vs RandomizedSearchCV
- Use ML pipelines to chain preprocessing + modeling

### ✅ Explainable AI (XAI)
- Use SHAP and LIME to explain black-box models
- Show feature contributions
- Improve model trust and debugging

---

## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- SHAP, LIME
- Jupyter Notebooks

---

## 🚀 How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/Machine_Learning.git

```bash
# Step 1: Go into the folder
cd Machine_Learning

# Step 2: Launch the Jupyter notebooks
jupyter notebook
