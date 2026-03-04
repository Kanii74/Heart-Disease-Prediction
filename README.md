# Heart Disease Prediction using Machine Learning

This repository contains the **Python implementation of a Machine Learning system for predicting heart disease in patients** using clinical attributes from the **UCI Heart Disease Dataset**.

The project explores multiple **Machine Learning classification models**, performs **Exploratory Data Analysis (EDA)**, handles **missing values using imputation techniques**, and evaluates the predictive performance of different algorithms.

The dataset used in this project is available at:

https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

---

# Overview

Heart disease is one of the leading causes of mortality worldwide. Early prediction and diagnosis can significantly improve patient outcomes and enable preventive healthcare.

Machine learning techniques allow us to analyze patient medical data and detect patterns that indicate the presence or absence of cardiovascular disease.

The goal of this project is to build a **predictive model that determines whether a patient is at risk of heart disease based on clinical parameters**.

---

# Dataset

The dataset used in this project is the **UCI Cleveland Heart Disease Dataset**, which contains clinical data for approximately **920 patients**.

Although the full database contains **76 attributes**, most machine learning studies use a subset of **14 important medical features**, which are also used in this project.

### Main Dataset Features

| Feature | Description |
|------|-------------|
| age | Age of the patient |
| sex | Gender of the patient |
| cp | Chest pain type |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar (>120 mg/dl) |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia status |
| target | Presence of heart disease |

---

# Target Variable

The original dataset contains **multiple levels of heart disease severity**:

| Value | Meaning |
|------|--------|
| 0 | No heart disease |
| 1 | Mild heart disease |
| 2 | Moderate heart disease |
| 3 | Severe heart disease |
| 4 | Critical heart disease |

For binary classification, the target variable is converted as:

- $0 \rightarrow$ No heart disease  
- $1,2,3,4 \rightarrow 1$ (Heart disease present)

Thus the final prediction task becomes a **binary classification problem**:

$$
y =
\begin{cases}
0 & \text{No Heart Disease} \\
1 & \text{Heart Disease Present}
\end{cases}
$$

---

# Data Preprocessing

Before training the machine learning models, several preprocessing steps were applied.

### 1 Handling Missing Values

The dataset contains missing values in multiple features.

Missing values were handled using a combination of:

- **Iterative Imputation**
- **Random Forest based prediction**
- **Mean imputation for continuous variables**

This ensures that the dataset remains complete and usable for model training.

---

### 2 Categorical Encoding

Categorical features such as:

- chest pain type
- ECG results
- thalassemia status

were converted into numerical form using **Label Encoding**.

---

### 3 Feature Scaling

Numerical features were normalized using **MinMax Scaling**:

$$
x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

This ensures all features lie within the range:

$$
0 \leq x_{scaled} \leq 1
$$

which improves the performance of many machine learning algorithms.

---

### 4 Polynomial Feature Engineering

To capture interactions between variables, **polynomial interaction features** were generated using:

$$
X_{poly} = \text{PolynomialFeatures}(X)
$$

with interaction terms up to degree 2.

---

# Exploratory Data Analysis

Several visualization techniques were used to understand the dataset:

- Histograms of numerical variables
- Frequency plots of categorical variables
- Box plots to detect outliers
- Pair plots to observe relationships between features
- Correlation heatmaps to identify dependencies

The correlation coefficient between two variables is defined as:

$$
r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$

where:

- $\text{Cov}(X,Y)$ is the covariance  
- $\sigma_X, \sigma_Y$ are standard deviations  

The value of $r$ lies between:

$$
-1 \le r \le 1
$$

---

# Machine Learning Models Implemented

Multiple supervised machine learning algorithms were implemented and evaluated.

### Logistic Regression

Logistic regression models the probability that a patient has heart disease.

The logistic function (sigmoid function) is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where

$$
z = w^T x + b
$$

The model predicts the probability:

$$
P(y=1|x) = \sigma(w^T x + b)
$$

If the probability exceeds a threshold (usually 0.5), the patient is classified as having heart disease.

---

### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees.

Each tree is trained on a random subset of the data using **bootstrap sampling**.

For classification, the final prediction is determined by **majority voting**:

$$
\hat{y} = \text{mode}(T_1(x), T_2(x), ..., T_n(x))
$$

where

- $T_i(x)$ is the prediction from the $i^{th}$ decision tree.

Random Forest reduces overfitting and improves generalization performance.

---

### Additional Models Used

The notebook also imports and experiments with several other algorithms:

- K-Nearest Neighbors
- Support Vector Machines
- Decision Trees
- Gradient Boosting
- AdaBoost
- XGBoost

Each model computes prediction accuracy for comparison.

---

# Model Training

The dataset was divided into training and testing sets:

- **70% training data**
- **30% testing data**

This was done using:

```
train_test_split()
```

Model performance was evaluated using:

- Accuracy score
- Confusion matrix
- Classification report

---

# Project Structure

```
Heart-Disease-Prediction
│
├── Code.ipynb
│
├── Heart_Disease_Prediction_Report.pdf
│
└── README.md
```

### Files

**Code.ipynb**

Contains the full machine learning workflow:

- data loading
- preprocessing
- EDA
- missing value imputation
- model training
- model evaluation

**Heart_Disease_Prediction_Report.pdf**

Detailed project report describing:

- background of heart disease
- machine learning methodology
- dataset description
- model explanations
- potential application as a health prediction app.

---

# Potential Application

The model developed in this project can serve as the core engine of a **Heart Disease Prediction Application**.

Such an application could allow users to:

- input their medical parameters
- receive a risk prediction
- obtain recommendations for medical consultation
- monitor cardiovascular health over time

This could help improve **early detection and preventive healthcare**.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly

---

# Disclaimer

This project is intended for **educational and research purposes only**.  
It should **not be used as a substitute for professional medical diagnosis**.
