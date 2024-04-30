#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


salary_ds=pd.read_csv("E:\Data Science Projects\Datasets/Salary_Data.csv")
salary_ds.head()


# In[3]:


#descriptive analysis
salary_ds.describe()


# In[5]:


#shape
salary_ds.shape


# In[6]:


#checking for null values
salary_ds.isnull().sum()


# In[9]:


#Scatter plot to show relationship
fig=px.scatter (data_frame=salary_ds, x="Salary", y="YearsExperience", size="YearsExperience", trendline="ols")
fig.show()


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x=np.asanyarray(salary_ds[["YearsExperience"]])
y=np.asanyarray(salary_ds[["Salary"]])

xtrain,xtest,ytrain, ytest= train_test_split (x,y, test_size=0.2, random_state=42)


# In[25]:


#model training
model=LinearRegression()
model.fit(xtrain, ytrain)


# In[26]:


#Salary prediction
predic_sal= float(input("Please enter the years of experience: "))

features=np.array([[predic_sal]])
print ("Your Salary will be = ", model.predict(features))


# In[30]:


from sklearn.metrics import r2_score
model.score(xtest,ytest)


# In[ ]:




