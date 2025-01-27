# Cardiovascular Disease Prediction using Machine Learning

## Overview
This project focuses on developing predictive models to assess the likelihood of cardiovascular disease (CVD) based on clinical and demographic factors. Using a dataset sourced from Kaggle, we implemented and evaluated multiple machine learning models to classify patients as at-risk or not.

### Problem Statement
Cardiovascular diseases are a leading cause of mortality worldwide. Early detection is critical to prevent severe outcomes. Machine learning enables predictive analytics by identifying patterns in large datasets, making it an effective tool for diagnosing CVD.

---

## Dataset
**Source:** [Kaggle: Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/)

### Description:
- **Input Features:** Age, Height, Weight, Gender, Blood Pressure, Cholesterol, Glucose Levels, Smoking, Alcohol Consumption, Physical Activity.
- **Target Variable:** Presence of CVD (0 = no disease, 1 = disease).
- **Dataset Size:** 70,000 samples, 12 features.

### Preprocessing:
- Handled missing values, duplicates, and outliers (e.g., blood pressure, BMI).
- Converted age from days to years.
- Standardized numerical features.
- No one-hot encoding for ordinal variables like cholesterol and glucose.

---

## Machine Learning Models
We implemented and compared the following models:

1. **Logistic Regression**
   - Hyperparameters: `C=1`, `max_iter=100`, `penalty='l2'`, `solver='liblinear'`
   - Regularization: L2

2. **Decision Tree**
   - Hyperparameters: `criterion='gini'`, `max_depth=5`, `min_samples_leaf=20`

3. **K-Nearest Neighbors (KNN)**
   - Hyperparameters: `n_neighbors=200`, `algorithm='auto'`, `leaf_size=40`

4. **Support Vector Machine (SVM)**
   - Hyperparameters: `kernel='rbf'`, `gamma=0.01`, `C=1.0`

### Hyperparameter Tuning:
Performed using GridSearchCV for all models. Most models exhibited minimal accuracy improvements (<1%) after tuning.

---

## Evaluation Metrics
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Cross-Validation:** 3-fold for hyperparameter tuning and evaluation.

### Results Summary:
| Model                 | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC-AUC |
|-----------------------|----------|---------------|---------------|------------|------------|---------------|---------------|---------|
| Logistic Regression   | 71.59%   | 0.79          | 0.74          | 0.78       | 0.65       | 0.73          | 0.70          | 0.7155  |
| Decision Tree         | 72.14%   | 0.69          | 0.78          | 0.82       | 0.62       | 0.75          | 0.69          | 0.7207  |
| KNN                   | 71.03%   | 0.69          | 0.74          | 0.77       | 0.65       | 0.73          | 0.69          | 0.7099  |
| SVM                   | 71.79%   | 0.70          | 0.74          | 0.77       | 0.67       | 0.73          | 0.70          | 0.7176  |

---

## Observations
- **Best Model:** Support Vector Machine (SVM) consistently achieved the best balance between accuracy and sensitivity.
- **Preprocessing Impact:** Outlier removal and BMI feature creation improved performance.
- **Feature Selection:** Attempted feature selection using Chi-Squared and ANOVA f-value scoring but observed performance degradation, so all features were retained.

---

## Limitations
1. **Dataset Quality:** Outliers and inconsistencies in the dataset impacted model performance.
2. **Feature Engineering:** While BMI improved results, other engineered features were less effective.
3. **Minimal Impact of Hyperparameter Tuning:** Most models showed negligible accuracy improvements.

---

## Technical Stack
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## Conclusion
This project demonstrates the effectiveness of machine learning for predicting cardiovascular disease. SVM emerged as the most effective model, striking a balance between precision and recall. Further improvements could focus on obtaining higher-quality data and exploring advanced feature engineering techniques.

---

## References
1. Evaluation Metrics: [Analytics Vidhya](https://analyticsvidhya.com/blog/2019/08/11-important-model-evaluation-error-metrics/)
2. Feature Selection: [Scikit-learn](https://scikit-learn.org/1.5/modules/feature_selection.html)
3. Dataset: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/)

