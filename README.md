# Salary-Prediction---Simple-Linear-Regression

A beginner-friendly machine learning project that predicts salary from years of experience using Simple Linear Regression.

---

## What This Project Does

Takes a person's years of experience as input and outputs a predicted salary. The model learns from 30 real employee records and draws the best-fit straight line through the data.

---

## Dataset

**Source:** [Salary Dataset — Kaggle](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression)

| Column | Role |
|--------|------|
| `YearsExperience` | Input (X) |
| `Salary` | Output (y) |

- 30 rows, no missing values
- Experience range: 1.2 – 10.6 years
- Salary range: ₹37,732 – ₹1,22,392

> **Note:** The CSV has an extra `Unnamed: 0` index column — it is loaded but not used in training.

---

## Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
```

---

## How It Works

**1. Load & Explore**
Read the CSV, check shape, view first 5 rows, and confirm zero missing values.

**2. Visualise**
Scatter plot of YearsExperience vs Salary — shows a clear upward linear trend.

**3. Prepare Data**
```python
X = df[["YearsExperience"]]   # input
y = df["Salary"]               # target
```

**4. Train / Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 24 rows → training | 6 rows → testing
```

**5. Train the Model**
```python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**6. Formula Learned**
```
Salary = 9423.82 × YearsExperience + 24380.20
```
Every extra year of experience adds approximately **₹9,424** to salary.

---

## Results

| Metric | Value |
|--------|-------|
| R² Score | **0.9024** |
| MAE | **₹6,286.45** |
| Slope (m) | 9423.82 |
| Intercept (c) | 24380.20 |

R² of **0.90** means the model explains **90.2%** of salary variation using experience alone.

---

## Charts Generated

| File | Description |
|------|-------------|
| `chart1_scatter_raw.png` | Raw data scatter plot |
| `chart2_regression_line.png` | Fitted regression line over training data |
| `chart3_actual_vs_predicted.png` | Side-by-side bar chart of actual vs predicted salaries |
| `chart4_residuals.png` | Residual plot — errors scattered around zero |

---

## Sample Prediction

```python
years = 6
new_person = pd.DataFrame([[years]], columns=["YearsExperience"])
model.predict(new_person)
# Output: ₹80,923.09
```

---

## Known Issue in the Notebook

The final summary `print` statement hard-codes `"R² ≈ 0.96"` and `"₹9,450"` — both are **incorrect** against the actual computed outputs (`R² = 0.9024`, slope = `9423.82`). The printed values should use the variables `r2` and `model.coef_[0]` instead of hardcoded strings.

---

## Limitation

Only one feature (experience) is used. Real salary also depends on location, job title, and skills. Adding more features with **Multiple Linear Regression** would improve accuracy.

---

## How to Run

1. Clone this repo and open `Salary_prediction_-_Simple_linear_regression.ipynb` in Jupyter or Google Colab
2. Upload `Salary_dataset.csv` to the same directory
3. Run all cells in order

---

## Author

**Raman** — B.Sc. (Hons.) Physics, Keshav Mahavidyalaya
