#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Import the libraries
import pandas as pd
import warnings
warnings.simplefilter('ignore')


# In[17]:


#load the data
shop_ds= pd.read_csv("E:\Data Science Projects\Datasets/shopping_data.csv")
shop_ds.head(10)


# In[6]:


shop_ds.columns


# In[7]:


#Define the shape of data
shop_ds.shape


# In[10]:


#reading data from a sing column
shop_ds["Annual Income (k$)"]


# In[14]:


#reading data from a single row
shop_ds.iloc[4]


# In[16]:


#reading data from a single cell
shop_ds['Age'].iloc[3]


# In[18]:


#reading data from multiple cells
shop_ds["Age"].iloc[6:10]


# In[19]:


#reading data from a pre-defined range
shop_ds.iloc[6:10]


# In[23]:


#getting all the statistical details of data
shop_ds.describe(include='all')


# In[24]:


#checking for null values
shop_ds.isnull().sum()


# In[35]:


# loading data with missing values
shop_missing=pd.read_csv("E:\Data Science Projects\Datasets/shopping_data_missingvalue.csv")
shop_missing.head()


# In[36]:


#Checking the msiing values
shop_missing.isnull().sum()


# In[39]:


# statistical summary of the data
shop_missing.describe()


# In[40]:


#reading the first 10 rows
shop_missing.head(10)


# In[41]:


# filling the null values using our mean and showing the results
fill_data= shop_missing.fillna(shop_missing.mean())
fill_data.head(10)


# In[43]:


#filling the null values using the median
data_filling_median=shop_missing.fillna(shop_missing.median())
data_filling_median.head(10)


# In[ ]:




