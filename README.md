# 🩺 Exploratory Data Analysis (EDA) on Diabetes Dataset

This project is a **beginner-friendly data analysis and machine learning notebook** that walks through understanding a diabetes dataset, performing basic **EDA (Exploratory Data Analysis)**, and building a simple **K-Nearest Neighbors (KNN)** classification model.

If you are new to **Python, Data Science, or Machine Learning**, this README explains *each step* of the notebook in simple terms.

---

## 📌 Project Objective

The main goals of this notebook are:

1. Load and understand the diabetes dataset
2. Explore the data using statistics and visualizations
3. Prepare the data for machine learning
4. Build and evaluate a KNN classification model

---

## 📂 Dataset Used

* **Dataset Name**: Pima Indians Diabetes Dataset
* **Source**: Kaggle
* **File**: `diabetes.csv`

### 🧾 Columns Description

| Column Name              | Meaning                        |
| ------------------------ | ------------------------------ |
| Pregnancies              | Number of times pregnant       |
| Glucose                  | Plasma glucose concentration   |
| BloodPressure            | Diastolic blood pressure       |
| SkinThickness            | Triceps skin fold thickness    |
| Insulin                  | 2-Hour serum insulin           |
| BMI                      | Body Mass Index                |
| DiabetesPedigreeFunction | Diabetes family history        |
| Age                      | Age of the patient             |
| Outcome                  | 0 = Non-diabetic, 1 = Diabetic |

---

## 🛠️ Libraries Used

The notebook uses the following Python libraries:

* **pandas** → Data handling and manipulation
* **numpy** → Numerical operations
* **matplotlib & seaborn** → Data visualization
* **scikit-learn** → Machine learning algorithms and evaluation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 📥 Step 1: Loading the Dataset

The dataset is loaded using `pandas.read_csv()`.

```python
data = pd.read_csv('diabetes.csv')
```

This converts the CSV file into a DataFrame for easy analysis.

---

## 👀 Step 2: Understanding the Data

### View First 5 Rows

```python
data.head()
```

Helps understand how the data looks.

### Statistical Summary

```python
data.describe()
```

Provides mean, min, max, and other statistics.

---

## ❓ Step 3: Checking Data Quality

### Dataset Info

```python
data.info()
```

Shows:

* Column names
* Data types
* Non-null counts

### Missing Values Check

```python
data.isnull().sum()
```

Confirms whether any missing (null) values exist.

---

## 📊 Step 4: Exploratory Data Analysis (EDA)

EDA helps understand patterns, trends, and relationships.

### Correlation Heatmap

```python
sns.heatmap(data.corr(), annot=True)
```

Shows relationships between features.

### Count Plot for Outcome

```python
sns.countplot(x='Outcome', data=data)
```

Displays number of diabetic vs non-diabetic patients.

### Distribution Plots

Used to visualize how values are spread for features like glucose, BMI, age, etc.

---

## ✂️ Step 5: Splitting Features and Target

```python
X = data.drop('Outcome', axis=1)
y = data['Outcome']
```

* **X** → Input features
* **y** → Target (diabetes result)

---

## 🔀 Step 6: Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

* 80% training data
* 20% testing data

---

## 🤖 Step 7: K-Nearest Neighbors (KNN) Model

### What is KNN?

KNN is a simple machine learning algorithm that:

* Looks at the **K nearest data points**
* Predicts based on majority class

---

### Choosing the Best K Value

The notebook tests different K values and stores accuracy scores.

```python
for k in range(1, 15):
    knn = KNeighborsClassifier(k)
```

A line plot is used to visualize training vs testing accuracy.

---

## 🏆 Step 8: Training Final Model

```python
knn = KNeighborsClassifier(13)
knn.fit(X_train, y_train)
```

The best K value (13) is chosen based on accuracy.

---

## 📈 Step 9: Model Evaluation

### Accuracy Score

```python
knn.score(X_test, y_test)
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### Classification Report

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

Includes:

* Precision
* Recall
* F1-score
* Support

---

## ✅ Final Outcome

* Successfully analyzed the diabetes dataset
* Built a KNN classifier
* Evaluated model performance using standard metrics

This notebook is a **great starting point** for beginners learning:

* Data analysis
* Visualization
* Basic machine learning

---

## 🚀 Next Improvements (Optional)

* Feature scaling (StandardScaler)
* Try other models (Logistic Regression, Random Forest)
* Handle zero values more carefully
* Hyperparameter tuning

---


