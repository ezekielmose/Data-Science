#!/usr/bin/env python
# coding: utf-8

# ### Loan Approval Prediction Model

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter ('ignore')


# In[2]:


loan_ds=pd.read_csv("E:\Data Science Projects\Datasets/loan_prediction.csv")
loan_ds.head()


# In[3]:


loan_ds.isnull().sum()


# In[4]:


loan_ds.describe()


# In[5]:


loan_ds.info()


# In[6]:


# fill the missing values
loan_ds['Gender'].fillna(loan_ds['Gender'].mode()[0],inplace=True)
loan_ds['Married'].fillna(loan_ds['Married'].mode()[0], inplace=True)
loan_ds['Dependents'].fillna(loan_ds['Dependents'].mode()[0],inplace=True)
loan_ds['Self_Employed'].fillna(loan_ds['Self_Employed'].mode()[0], inplace=True)


# In[7]:


loan_ds.isnull().sum()


# In[8]:


loan_ds['LoanAmount'].fillna(loan_ds['LoanAmount'].median(), inplace=True)
loan_ds['Loan_Amount_Term'].fillna(loan_ds['Loan_Amount_Term'].mode()[0],inplace=True)
loan_ds['Credit_History'].fillna(loan_ds['Credit_History'].mode()[0], inplace=True)


# In[9]:


loan_ds.isnull().sum()


# In[10]:


import plotly.express as px
loan_status_count= loan_ds['Loan_Status'].value_counts()
pie_c=px.pie(loan_status_count, names=loan_status_count.index, title="Loan Approval Status")
pie_c.show()


# In[11]:


# distribution of gender counts using bargraph
gender_counts= loan_ds['Gender'].value_counts()
gender_bar=px.bar(gender_counts, x=gender_counts.index, y=gender_counts.values, title="Gender Distribution")
gender_bar.show()
gender_counts


# In[12]:


# distribution of marital status counts using bargraph
marital_status_counts= loan_ds['Married'].value_counts()
marital_bar=px.bar(marital_status_counts, x=marital_status_counts.index, y=marital_status_counts.values, title="Marital Status  Distribution")
marital_bar.show()
marital_status_counts


# In[13]:


# distribution of marital status counts using bargraph
education_counts= loan_ds['Education'].value_counts()
education_bar=px.bar(education_counts, x=education_counts.index, y=education_counts.values, title="Education Status  Distribution")
education_bar.show()
education_counts


# In[14]:


self_employed_counts= loan_ds['Self_Employed'].value_counts()
employment_bar=px.bar(self_employed_counts, x=self_employed_counts.index, y=self_employed_counts.values, title="Self Employed Status  Distribution")
employment_bar.show()
self_employed_counts


# In[15]:


# Distribution of the applicants income using histogram
app_income= px.histogram (loan_ds, x='ApplicantIncome', title='The Applicant Income')
app_income.show()


# In[16]:


# relationship between income of the loan applicant and the loan status
loan_inc_status=px.box(loan_ds, x='Loan_Status', y='ApplicantIncome', color='Loan_Status',title = 'Relationship between loan aplicant income and loan status')
loan_inc_status.show()


# In[17]:


#removing the outliers
#Calculate the interquatile range
Q1=loan_ds['ApplicantIncome'].quantile(0.25)
Q3=loan_ds['ApplicantIncome'].quantile(0.75)
IQR=Q3-Q1

#define lower and upper bounds of the outliers
lower_bound = Q1-1.5*IQR
upper_bound = Q3-1.5+IQR

#Remove the outliers
loan_ds=loan_ds[(loan_ds['ApplicantIncome']>=lower_bound) & loan_ds['ApplicantIncome']<=upper_bound]


# In[18]:


# relationship between a income and te loan co apllicant
loan_coapp_income= px.violin(loan_ds, x='Loan_Status', y='CoapplicantIncome', color = 'Loan_Status', title='relationship between loan co_applicant and the income')
loan_coapp_income.show()


# In[19]:


# remove outliers from the coapplicant column

#Find the IQR
Q1= loan_ds['CoapplicantIncome'].quantile(0.25)
Q3 = loan_ds ['CoapplicantIncome'].quantile(0.75)
IQR=Q3-Q1

#upper and lowwr bounds
lower_bound=Q1-1.5*IQR
upper_bound=Q3+15*IQR

#remove outliers
loan_ds=loan_ds[(loan_ds['CoapplicantIncome']>=lower_bound)& loan_ds['CoapplicantIncome']<=upper_bound]


# In[20]:


loan_status_amount=px.box(loan_ds, x='Loan_Status', y='LoanAmount', color='Loan_Status', title = 'Loan Status vs Loan Amount')
loan_status_amount.show()


# In[21]:


#Loan status vs Propert area
loan_status_pro=px.histogram(loan_ds, x='Credit_History', barmode='group',color='Loan_Status', title='loan status vs Property area')
loan_status_pro.show()


# In[22]:


loan_ds.head()


# In[55]:



# Converting the categorical columns to numerical columns
categoric_col=['Gender','Married','Dependents', 'Education', 'Self_Employed', 'Property_Area']
df=pd.get_dummies(loan_ds, columns=categoric_col)

#Split the dataset into features X and target Y
x=loan_ds.drop('Loan_Status', axis=1)
y=loan_ds['Loan_Status']

#Split the dataset into training and testing sets
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)

#scaling the numerical columns using standard scalar
from sklearn.preprocessing import StandardScaler
sscaler=StandardScaler()
num_cols=['LoanAmount','CoapplicantIncome','Loan_Amount_Term', 'Credit_History'  ]
x_train[num_cols]=sscaler.fit_transform(x_train[num_cols])
x_test[num_cols]=sscaler.transform(x_test[num_cols])

from sklearn.svm import SVC
model=SVC(random_state=42)
model.fit(x_train,y_train)


# In[ ]:




