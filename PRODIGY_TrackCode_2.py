#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd 
import numpy as np 
df=pd.read_csv("C:\\Users\\Smit\\OneDrive\\Desktop\\DATASET\\train.csv")
df.head(10)


# In[11]:


df.isna().sum()


# In[12]:


df.describe()


# In[ ]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna(df['Cabin'].median(), inplace=True)
df.isnull().sum()


# In[17]:


df.isnull().sum()


# In[19]:


df.drop(columns=['Cabin'], inplace=True)
df.head()
df.isnull().sum()


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Survived', data=df, palette='coolwarm')
plt.title("Survival Count")
plt.show()
plt.hist(df['Age'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[22]:


sns.barplot(x='Sex', y='Survived', data=df, palette='coolwarm')
plt.title("Survival Rate by Gender")
plt.show()
sns.barplot(x='Pclass', y='Survived', data=df, palette='coolwarm')
plt.title("Survival Rate by Class")
plt.show()


# In[23]:


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[26]:


features = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Embarked']
X = df[features]
y = df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape


# In[28]:


df.select_dtypes(include=['object']).columns


# In[33]:


columns_to_drop = ['Name', 'Ticket']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print(df.columns)


# In[36]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


# In[37]:


print(df.dtypes)


# In[38]:


df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], drop_first=True)


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X = df.drop(columns=['Survived']) 
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model training successful")


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




