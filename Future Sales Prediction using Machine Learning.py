#!/usr/bin/env python
# coding: utf-8

# ## Future Sales Prediction using Machine Learning

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter ('ignore')


# In[3]:


sales_ds = pd.read_csv ("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
sales_ds.head()


# In[5]:


sales_ds.isnull().sum()


# In[6]:


sales_ds.info()


# In[7]:


sales_ds.describe()


# In[12]:


import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = sales_ds, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()


# In[13]:


figure = px.scatter(data_frame = sales_ds, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()


# In[15]:


figure = px.scatter(data_frame = sales_ds, x="Sales",
                    y="Radio", size="Radio", trendline="ols")
figure.show()


# In[16]:


correlation=sales_ds.corr()
correlation['Sales'].sort_values(ascending=False)


# In[17]:


x = np.array(sales_ds.drop(["Sales"], 1))
y = np.array(sales_ds["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42) 
                                              


# In[18]:


model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[ ]:




