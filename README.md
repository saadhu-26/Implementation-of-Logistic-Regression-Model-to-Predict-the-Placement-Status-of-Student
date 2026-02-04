# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the given placement dataset and preprocess the data by encoding categorical values.

2.Split the dataset into training and testing sets.

3.Train the Logistic Regression model using the training data.

4.Predict the placement status and evaluate the model using accuracy.  

## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:SAADHANA A 
RegisterNumber:25018432  
*/
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# Label Encoding for categorical columns
le = LabelEncoder()

categorical_columns = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
    "status"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Features (X) and Target (y)
X = df.drop(["status", "salary", "sl_no"], axis=1)
y = df["status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Output:

<img width="551" height="494" alt="Screenshot 2026-02-04 091747" src="https://github.com/user-attachments/assets/83b4bf32-5dbe-4489-abd9-e0207beb9da9" />

## Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
