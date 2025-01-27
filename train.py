import pandas as pd
import numpy as np
from math import floor
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import model_selection

np.random.seed(42)

# Read Data and Convert to Dataframe
df = pd.read_csv('cardio_train.csv', sep=';')

# function to convert age from days to years
def convert_to_years(x):
    return floor(x/365)


# Drop Duplicates and Missing Values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("missing values left in dataset (all values should equal zero) \n", df.isna().sum())

# drop id
df.drop(columns=['id'], inplace=True)

# Convert Days to Years for age
df['age'] = df['age'].apply(convert_to_years)

# BMI Calculation and Filtering: 10 < BMI < 60 
BMI = round(df['weight'] / ((df['height'] / 100) ** 2), 1)
df = df[(BMI >= 10) & (BMI <= 60)]  # Filter BMI values

# convert gender to binary
df['gender'] = df['gender'].map({1: 0, 2: 1})

# IQR
aphi = df['ap_hi'].sort_values()

Q1 = aphi.quantile(0.25)
Q3 = aphi.quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# after running on initial data got upper limit of 170 and lower of 90
print("upper bound\n", upper_bound)
print("lower bound \n", lower_bound)
print("num of people with outlier levels of blood pressure (lower bound)",
      df[df['ap_hi'] < lower_bound].shape[0])
print("num of people with outlier levels of blood pressure (high)",
      df[df['ap_hi'] > upper_bound].shape[0])
# Rows outside the IQR range
outliers = df[(df['ap_hi'] < lower_bound) | (df['ap_hi'] > upper_bound)]
print(f"Percentage of outliers: {(outliers.shape[0]/70000)*100}%")
# Rows within the IQR range
df = df[(df['ap_hi'] >= lower_bound) & (df['ap_hi'] <= upper_bound)]
print(f"Number of rows within bounds: {df.shape[0]}")


# IQR Again for diastolic blood
aplo = df['ap_lo'].sort_values()
Q1 = aplo.quantile(0.25)
Q3 = aplo.quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# after running on initial data got upper limit of 170 and lower of 90
print("upper bound\n", upper_bound)
print("lower bound \n", lower_bound)
print("num of people with outlier levels of blood pressure (lower bound)",
      df[df['ap_lo'] < lower_bound].shape[0])
print("num of people with outlier levels of blood pressure (high)",
      df[df['ap_lo'] > upper_bound].shape[0])

# Rows outside the IQR range
outliers = df[(df['ap_lo'] < lower_bound) | (df['ap_lo'] > upper_bound)]
print(f"Percentage of outliers: {(outliers.shape[0]/70000)*100}%")
# Rows within the IQR range
df = df[(df['ap_lo'] >= lower_bound) & (df['ap_lo'] <= upper_bound)]
print(f"Number of rows within bounds: {df.shape[0]}")

# Lets plot our blood pressures
fig, axes = plt.subplots(3, 4, figsize=(12, 10))

axes = axes.flatten()
axes[0].set_title("Systolic B.P. Distribution")
sns.histplot(df['ap_hi'], ax=axes[0])
axes[1].set_title("Diastolic B.P. Distribution")
sns.histplot(df['ap_lo'], ax=axes[1])
axes[2].set_title("Gender Distribution")
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(["Female", "Male"])
sns.histplot(df['gender'], ax=axes[2], discrete=True)
axes[3].set_title("Age Distribution")
sns.histplot(df['age'], ax=axes[3])
axes[4].set_title("Weight Distribution")
sns.histplot(df['weight'], ax=axes[4])
axes[5].set_title("Cholesterol Distribution")
axes[5].set_xticks([1, 2, 3])
axes[5].set_xticklabels(["normal", "abv_normal", "well_abv_normal"])
sns.histplot(df['cholesterol'], ax=axes[5], discrete=True)
axes[6].set_title("Glucose Distribution")
axes[6].set_xticks([1, 2, 3])
axes[6].set_xticklabels(["normal", "abv_normal", "well_abv_normal"])
sns.histplot(df['gluc'], ax=axes[6], discrete=True)
axes[7].set_title("Smokers Distribution")
axes[7].set_xticks([0, 1])
axes[7].set_xticklabels(['Non-Smoker', 'Smoker'])
sns.histplot(df['smoke'], ax=axes[7], discrete=True)
axes[8].set_title("Alcohol Distribution")
axes[8].set_xticks([0, 1])
axes[8].set_xticklabels(['Non-Drinker', 'Drinker'])
sns.histplot(df['alco'], ax=axes[8], discrete=True)
axes[9].set_title("Physically Active Distribution")
axes[9].set_xticks([0, 1])
axes[9].set_xticklabels(['Not Active', 'Active'])
sns.histplot(df['active'], ax=axes[9], discrete=True)
plt.tight_layout()
plt.show()

# split data
Y = df['cardio']
df.drop(columns=['cardio'], inplace=True)
X = df

# We tried to use chi2 and f_classif to select features but it seems 
# that the features are not linearly separable and 
# we were getting worse accuracy after removing some features 
# so we decided to keep all features


# #Numerical Features
# numerical_X = X[['age','height','weight','ap_hi','ap_lo']]

# f_values, p_values = f_classif(numerical_X, Y)

# # Display results
# anova_results = pd.DataFrame({
#     "Feature": numerical_X.columns,
#     "F-Value": f_values,
#     "P-Value": p_values
# }).sort_values(by="F-Value", ascending=False)

# print(anova_results)

# #Categorial
# categorial_X = X[['gender','cholesterol','gluc','smoke','active','alco']]

# chi2_scores, p_values = chi2(categorial_X, Y)

# # Display results
# chi2_results = pd.DataFrame({
#     "Feature": categorial_X.columns,
#     "Chi2-Score": chi2_scores,
#     "P-Value": p_values
# }).sort_values(by="Chi2-Score", ascending=False)

# print(chi2_results)

# #removing some features with low chi/f-scores and high p vals
# X.drop(columns=['height'], inplace=True)

# print("Kept Features",X.columns)


# 80/20 split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True)


def get_data():
    return X_train, Y_train, X_test, Y_test
