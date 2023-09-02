#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


url = "C:/test.csv"  # Update the path
data = pd.read_csv(url)


# In[3]:


print(data.head())
print(data.info())
print(data.describe())


# In[4]:


# Pairplot for general overview
sns.pairplot(data, hue='total_day_minutes')  # Assuming 'international_plan' is the target column

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Distribution of Churn
plt.figure(figsize=(6, 4))
sns.countplot(data['total_day_minutes'])  # Assuming 'international_plan' is the churn indicator
plt.title("Churn Distribution")
plt.show()


# In[5]:


numerical_features = ['total_day_calls', 'total_day_charge', 'total_eve_minutes']  # Update with your features
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()


# In[6]:


categorical_features = ['state', 'international_plan', 'voice_mail_plan']  # Update with your features
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=feature, y='total_eve_calls', data=data)
    plt.title(f"{feature} vs total_eve_calls")
    plt.xticks(rotation=45)
    plt.show()


# In[7]:


print(data.head())


# In[18]:


import pandas as pd

# Assuming you have a DataFrame named 'df' and want to delete columns 'column1' and 'column2'
columns_to_delete = ['area_code','state']

# Use the drop method to delete the specified columns
data.drop(columns=columns_to_delete, inplace=True)

# The 'inplace=True' argument modifies the DataFrame in place, so you don't need to reassign it.


# In[19]:


print(data.info())


# In[11]:


print(data.isnull().sum())
# Depending on the columns with missing values, you can use methods like imputation.


# In[20]:


from sklearn.model_selection import train_test_split

X = data.drop(columns=["total_intl_calls"])
y = data["total_intl_calls"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression(max_iter=1000)# Import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Preprocess the categorical variables using one-hot encoding
#data = pd.get_dummies(data, columns=["Geography", "Gender"])

# Split into features (X) and target (y)
X = data.drop("number_customer_service_calls", axis=1)
y = data["number_customer_service_calls"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    #"Decision Tree": DecisionTreeClassifier(),
    #"Random Forest": RandomForestClassifier(),
    #"Support Vector Machine": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Specify 'weighted' for multiclass
    recall = recall_score(y_test, y_pred, average='weighted')  # Specify 'weighted' for multiclass
    f1 = f1_score(y_test, y_pred, average='weighted')  # Specify 'weighted' for multiclass
    
    results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}

# Display evaluation results
print("Model Evaluation Results:")
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['Accuracy']:.2f}")
    print(f"Precision: {metrics['Precision']:.2f}")
    print(f"Recall: {metrics['Recall']:.2f}")
    print(f"F1 Score: {metrics['F1']:.2f}")
    print()


# In[24]:


print(data.head())


# In[35]:


best_model = max(results, key=lambda k: results[k]["F1"])
print("Best Performing Model:", best_model)


# In[50]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data
input_data = (175, 258, 587, 123, 456, 258, 236, 147, 159, 368, 157, 147, 142, 124, 152)

# Changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Assuming you have a DataFrame named 'data' and want to predict 'number_customer_service_calls'
# Replace 'X' and 'y' with your actual features and target variable
X = data.drop("number_customer_service_calls", axis=1)
y = data["number_customer_service_calls"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Logistic Regression model# Create and train a Logistic Regression model with increased max_iter
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)


# Now, you can make predictions with the trained model
prediction = logistic_regression.predict(input_data_reshaped)

if prediction[0] == 0:
    print("Churn: No")
else:
    print("Churn: Yes")


# In[ ]:




