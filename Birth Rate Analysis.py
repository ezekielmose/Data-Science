#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set() 
import warnings
warnings.simplefilter('ignore')


# In[15]:


births=pd.read_csv("E:\Data Science Projects\Datasets/births.csv")
births.tail(8)


# In[14]:


births.isnull().sum()


# In[3]:


births1= births.fillna(method='ffill')
births2= births1.fillna(method='bfill')
births2['day'] = births2['day'].astype(int)
births2.head()


# In[4]:


births2.isnull().sum()


# In[5]:


# create a column decade
births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
births.head()


# In[6]:


birth_decade = births.pivot_table('births', index='decade',columns='gender', aggfunc='sum') 
birth_decade.plot() 
plt.ylabel("Total births per year") 
plt.show()


# In[7]:


#removing outliers using sigma clipping
quantiles=np.percentile(births['births'],[25,50,70])
ds=quantiles[1]
sig=0.74* (quantiles[1]-quantiles[0])


# In[8]:


births = births.query('(births > @ds - 5 * @sig) & (births < @ds + 5 * @sig)')
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')

births['dayofweek'] = births.index.dayofweek


# In[9]:


births.pivot_table('births', index='dayofweek',
                    columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day');
plt.show()


# In[10]:


births_month = births.pivot_table('births', [births.index.month, births.index.day])
print(births_month.head())

births_month.index = [pd.datetime(2012, month, day)
                      for (month, day) in births_month.index]
print(births_month.head())


# In[11]:


fig, ax = plt.subplots(figsize=(12, 4))
births_month.plot(ax=ax)
plt.show()


# In[ ]:




