#!/usr/bin/env python
# coding: utf-8

# ### Real Estate Price Prediction ML Projects

# In[6]:


#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

estate_ds=pd.read_csv("E:\Data Science Projects\Datasets/Real_Estate.csv")
estate_ds.head(10)


# In[4]:


estate_ds.info()


# In[5]:


estate_ds.describe()


# In[14]:


estate_ds.isnull().sum()


# In[13]:


# set the stle of the plot
sns.set_style('whitegrid')

#creating histograms for the numerical columns\

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
fig.suptitle ("Histogram of the real estate data" , fontsize=12)

cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude', 'House price of unit area']

for i,col in enumerate(cols):
    sns.histplot(estate_ds[col], kde=True, ax=axes [i//2, i%2])
    axes [i//2, i%2].set_title(col)
    axes [i//2, i%2].set_xlabel('')
    axes [i//2, i%2].set_ylabel('')
    
plt.show()


# In[18]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle ("Scatter Plot" , fontsize=12)

sns.scatterplot(data=estate_ds, x='House age', y='House price of unit area', ax= axes[0,0] )
sns.scatterplot(data=estate_ds, x='Distance to the nearest MRT station', y='House price of unit area', ax= axes[0,1] )
sns.scatterplot(data=estate_ds, x='Number of convenience stores', y='House price of unit area', ax= axes[1,0] )
sns.scatterplot(data=estate_ds, x='Latitude', y='House price of unit area', ax= axes[1,1] )

plt.show()


# In[19]:


# Correlation Matrix
corr_mat = estate_ds.corr()

#plotting the correlation matrx
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", linewidth = .5)
plt.title ("Correlation Matrix")
plt.show()


# In[20]:


corr_mat


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

#Select feartures and target variables
features = ["Distance to the nearest MRT station", "Number of convenience stores", "Latitude", "Longitude"]
target= "House price of unit area"

X= estate_ds[features]
y= estate_ds[target]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initializing the model
model= LinearRegression()

#training the model
model.fit(X_train, y_train)


# In[34]:


# Visualizing the model

lr_predic = model.predict(X_test)

#visualization

plt.scatter(lr_predic, y_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()


# In[ ]:




